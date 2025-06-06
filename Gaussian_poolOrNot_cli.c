/*
 *  Author: Paul Horton
 *  Copyright: Paul Horton 2021, All rights reserved.
 *  Created: 20211201
 *  Updated: 20240603
 *  Licence: GPLv3
 *  Description: Simple demonstration of a Bayesian way to guess at the number of components
 *               behind a sample of numerical data.
 *  Compile:  gcc -Wall -O3 -o Gaussian_poolOrNot Gaussian_poolOrNot.c GSLfun.c -lgsl -lgslcblas -lm
 *  Environment: $GSL_RNG_SEED
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "GSLfun.h"
#include "thread_setup.h"
/* ───────────  Global definitions and variables  ────────── */
#define USE_LOG_PROB  // Use log probabilities, a little slower but can handle large data_n
#define PRINT_RESULTS_ONLY

int data_n = 25;
int dataset_n = 10;
int cdf_gauss_n = 20;
int cdf_gamma_n = 10;
int cdf_jbeta_n = 40;

double *data;
double *cdfInv_Gauss;
double *cdfInv_gamma;
double *cdfInv_JBeta;


typedef struct {
  double mixCof;
  Gauss_params Gauss1;
  Gauss_params Gauss2;
} Gauss_mixture_params;


Gauss_params mu_prior_params = {0.0, 4.0};
double sigma_prior_param_a = 0.5;
double sigma_prior_param_b = 2.0;

static uint sampleRepeatNum = 2000000;

#ifdef USE_THREADS
static pthread_key_t rng_key;

static void init_rng(int thread_idx) {
  gsl_rng_env_setup();
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  unsigned long seed;
  if (getenv("GSL_RNG_SEED")) {
    seed = atoi(getenv("GSL_RNG_SEED")) + thread_idx;
  } else {
    seed = DEFAULT_RNG_SEED + thread_idx;  // Reproducible per-thread seed
  }
  gsl_rng_set(r, seed);
  pthread_setspecific(rng_key, r);
}
#endif


/* ───────────  Functions to help summarize or dump the data  ────────── */

double data_sample_mean() {
  double mean = 0.0;
  for (uint i = 0; i < data_n; ++i) {
    mean += data[i];
  }
  return mean / (double) data_n;
}

double data_sample_variance() {
  double mean = data_sample_mean();
  double var = 0.0;
  for (uint i = 0; i < data_n; ++i) {
    double diff = data[i] - mean;
    var += diff * diff;
  }
  return var / (double) data_n;
}

// ternary CMP function for use with qsort
int CMPdata(const void *arg1, const void *arg2) {
  double *a = (double *) arg1;
  double *b = (double *) arg2;
  return (*a > *b) - (*a < *b);
}


#ifdef USE_LOG_PROB
// Returns log(Σ(e^log_probs)))
double logsumexp(double *log_probs, int n) {
  double max_log = log_probs[0];
  for (int i = 1; i < n; ++i)
    if (isnan(max_log) || !isfinite(max_log) || log_probs[i] > max_log)
      max_log = log_probs[i];

  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    if (!isnan(log_probs[i]) && isfinite(log_probs[i]))
      sum += exp(log_probs[i] - max_log);

  return max_log + log(sum);
}
#endif


/* ───────────  Functions used for sampling/generating data   ────────── */

Gauss_params prior_Gauss_params_sample() {
  Gauss_params params;
  params.mu = GSLfun_ran_gaussian(mu_prior_params);
  params.sigma = sigma_of_precision(
      GSLfun_ran_gamma(sigma_prior_param_a, sigma_prior_param_b));
  return params;
}

Gauss_mixture_params prior_Gauss_mixture_params_sample() {
  Gauss_mixture_params params;
  params.mixCof = GSLfun_ran_beta_Jeffreys();
  params.Gauss1 = prior_Gauss_params_sample();
  params.Gauss2 = prior_Gauss_params_sample();
  return params;
}

void data_generate_1component(Gauss_params params) {
  for (uint i = 0; i < data_n; ++i) {
    data[i] = GSLfun_ran_gaussian(params);
  }
}

void data_generate_2component(Gauss_mixture_params params) {
  for (uint i = 0; i < data_n; ++i) {
    data[i] = GSLfun_ran_gaussian(
        gsl_ran_flat01() < params.mixCof ? params.Gauss1 : params.Gauss2);
  }
}


/* ───────────  Numerical integration precomputation  ────────── */

//  Precompute the cumulative probabilities of μ and σ discrete values.
//  The probabilities depend on the current prior_params values
void cdfInv_precompute() {
  cdfInv_Gauss = malloc(cdf_gauss_n * sizeof(double));
  cdfInv_gamma = malloc(cdf_gamma_n * sizeof(double));
  cdfInv_JBeta = malloc(cdf_jbeta_n * sizeof(double));
  double x;
  // Since Normal range is unbounded, precompute cdfInv for vals:  ¹⁄₍ₙ₊₁₎...ⁿ⁄₍ₙ₊₁₎
  for (uint i = 0; i < cdf_gauss_n; ++i) {
    x = (i + 1) / (double) (1 + cdf_gauss_n);
    cdfInv_Gauss[i] =
        gsl_cdf_gaussian_Pinv(x, mu_prior_params.sigma) + mu_prior_params.mu;
  }
  for (uint i = 0; i < cdf_gamma_n; ++i) {
    x = i / (double) (cdf_gamma_n);
    cdfInv_gamma[i] =
        gsl_cdf_gamma_Pinv(x, sigma_prior_param_a, sigma_prior_param_b);
    //printf( "cdfInv_Gamma[%u]= %g\n", i, cdfInv_gamma[i] );
  }
  for (uint i = 0; i < cdf_jbeta_n; ++i) {
    // By symmetry, only need Beta values for p ≦ 0.5.  For example p=0.8, is the same p=0.2 with Gauss components swapped.
    x = 0.5 * i / (double) (cdf_jbeta_n);
    cdfInv_JBeta[i] = gsl_cdf_beta_Pinv(x, 0.5, 0.5);
    //printf( "cdfInv_JBeta[%u]= %g\n", i, cdfInv_JBeta[i] );
  }
}


/* ───────────  Probability Computations on the Data  ────────── */

double ***prob_data_1gauss = NULL;


// Precompute the probabilities for a data point given a Gaussian model
void init_prob_data_1gauss() {
  if (prob_data_1gauss == NULL) {
    prob_data_1gauss = malloc(cdf_gauss_n * sizeof(double **));
    for (uint m = 0; m < cdf_gauss_n; ++m) {
      prob_data_1gauss[m] = malloc(cdf_gamma_n * sizeof(double *));
      for (uint s = 0; s < cdf_gamma_n; ++s) {
        prob_data_1gauss[m][s] = malloc(data_n * sizeof(double));
      }
    }
  }
  for (uint m = 0; m < cdf_gauss_n; ++m) {
    double mu = cdfInv_Gauss[m];
    for (uint s = 0; s < cdf_gamma_n; ++s) {
      double sigma = sigma_of_precision(cdfInv_gamma[s]);
      Gauss_params cur_params = {mu, sigma};
      for (uint i = 0; i < data_n; ++i) {
        prob_data_1gauss[m][s][i] =
            GSLfun_ran_gaussian_pdf(data[i], cur_params);
      }
    }
  }
}


// Return Ｐ[D|μ,σ]
double prob_data_given_1Gauss(const Gauss_params params) {
#ifndef USE_LOG_PROB
  double prob = 1.0;
  for (uint d = 0; d < data_n; ++d) {
    prob *= GSLfun_ran_gaussian_pdf(data[d], params);
  }
  return prob;
#else
  double log_prob = 0.0;
  for (uint i = 0; i < data_n; ++i) {
    log_prob += log(GSLfun_ran_gaussian_pdf(data[i], params));
  }
  return log_prob;
#endif
}


// Return Ｐ[D|m,μ₁,σ₁,μ₂,σ₂]
double prob_data_given_2Gauss(const double mixCof, const Gauss_params Gauss1,
                              const Gauss_params Gauss2) {
#ifndef USE_LOG_PROB
  double prob = 1.0;
  for (uint i = 0; i < data_n; ++i) {
    prob *= (1 - mixCof) * GSLfun_ran_gaussian_pdf(data[i], Gauss2) +
            mixCof * GSLfun_ran_gaussian_pdf(data[i], Gauss1);
  }
  return prob;
#else
  double log_prob = 0.0;
  for (uint i = 0; i < data_n; ++i) {
    double prob1 = GSLfun_ran_gaussian_pdf(data[i], Gauss1);
    double prob2 = GSLfun_ran_gaussian_pdf(data[i], Gauss2);
    log_prob += log((1 - mixCof) * prob2 + mixCof * prob1);
  }
  return log_prob;
#endif
}


// Return maximum likelihood of the data using a single Gaussian
double data_Gauss1_maxLikelihood() {
  Gauss_params params = {data_sample_mean(), data_sample_variance()};
  return prob_data_given_1Gauss(params);
}


#ifdef USE_THREADS
/* ──────── Parallel summing for cdf_gauss_n × cdf_gauss_n work  ──────── */

typedef struct {
  uint index;  // Flattened index (m1 * cdf_gauss_n + m2)
  double result;
} JobResult;

#define MAX_JOBS (cdf_gauss_n)
#define MAX_JOBS2 (cdf_gauss_n * cdf_gauss_n)

// static JobResult job_results[MAX_JOBS2];
static JobResult *job_results;
static pthread_mutex_t job_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint job_index = 0;

void *thread_worker_1Gauss_sum(void *arg) {
  while (1) {
    pthread_mutex_lock(&job_mutex);
    if (job_index >= MAX_JOBS) {
      pthread_mutex_unlock(&job_mutex);
      break;
    }
    uint m = job_index++;
    pthread_mutex_unlock(&job_mutex);
#ifndef USE_LOG_PROB
    double sum = 0.0;
    for (uint s = 0; s < cdf_gamma_n; ++s) {
      double sigma = sigma_of_precision(cdfInv_gamma[s]);
      Gauss_params cur_params = {mu, sigma};
      sum += prob_data_given_1Gauss(cur_params);
    }
    job_results[m].result = sum;
#else
    double *log_prob = malloc(cdf_gamma_n * sizeof(double));
    for (uint s = 0; s < cdf_gamma_n; ++s) {
      log_prob[s] = 0;
      for (uint i = 0; i < data_n; ++i) {
        log_prob[s] += log(prob_data_1gauss[m][s][i]);
      }
    }
    job_results[m].result = logsumexp(log_prob, cdf_gamma_n);
    free(log_prob);
#endif
    job_results[m].index = m;
  }

  return NULL;
}


double data_prob_1component_bySumming_parallel() {
  pthread_t threads[NUM_THREADS];

  job_index = 0;  // Reset shared index

  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_create(&threads[i], NULL, thread_worker_1Gauss_sum,
                   (void *) (intptr_t) i);
  }

  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }

#ifndef USE_LOG_PROB
  double total = 0.0;
  for (uint i = 0; i < MAX_JOBS; ++i) {
    total += job_results[i].result;
  }
  return total / (double) (cdf_gauss_n * cdf_gamma_n);
#else
  double log_probs[MAX_JOBS];
  for (uint i = 0; i < MAX_JOBS; ++i) {
    log_probs[i] = job_results[i].result;
  }
  if (MAX_JOBS == 0) {
    return 0.0;
  }
  double log_avg_prob =
      logsumexp(log_probs, MAX_JOBS) - log(cdf_gauss_n) - log(cdf_gamma_n);
  return log_avg_prob;
#endif
}


void *thread_worker_2Gauss_sum(void *arg) {
#ifdef USE_LOG_PROB
  double *log_prob =
      malloc(cdf_gamma_n * cdf_gamma_n * cdf_jbeta_n * sizeof(double));
#endif
  while (1) {
    pthread_mutex_lock(&job_mutex);
    if (job_index >= MAX_JOBS2) {
      pthread_mutex_unlock(&job_mutex);
      break;
    }
    uint idx = job_index++;
    pthread_mutex_unlock(&job_mutex);

    uint m1 = idx / cdf_gauss_n;
    uint m2 = idx % cdf_gauss_n;

    double sum = 0.0;
    size_t idx2 = 0;
    for (uint s1 = 0; s1 < cdf_gamma_n; ++s1) {
      for (uint s2 = 0; s2 < cdf_gamma_n; ++s2) {
        for (uint mi = 0; mi < cdf_jbeta_n; ++mi) {
          double mixCof = cdfInv_JBeta[mi];
          log_prob[idx2] = 0;
          for (uint i = 0; i < data_n; ++i) {
            double prob1 = prob_data_1gauss[m1][s1][i];
            double prob2 = prob_data_1gauss[m2][s2][i];
            log_prob[idx2] += log((1 - mixCof) * prob2 + mixCof * prob1);
          }
          idx2++;
        }
      }
    }

#ifdef USE_LOG_PROB
    sum = logsumexp(log_prob, cdf_gamma_n * cdf_gamma_n * cdf_jbeta_n);
#endif
    job_results[idx].index = idx;
    job_results[idx].result = sum;
  }
#ifdef USE_LOG_PROB
  free(log_prob);
#endif
  return NULL;
}


double data_prob_2component_bySumming_parallel() {
  pthread_t threads[NUM_THREADS];

  job_index = 0;  // Reset shared index

  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_create(&threads[i], NULL, thread_worker_2Gauss_sum, NULL);
  }

  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }

#ifndef USE_LOG_PROB
  double total = 0.0;
  for (uint i = 0; i < MAX_JOBS2; ++i) {
    total += job_results[i].result;
  }
  return total / (double) (cdf_gauss_n * cdf_gauss_n * cdf_gamma_n *
                           cdf_gamma_n * cdf_jbeta_n);
#else
  double log_probs[MAX_JOBS2];
  for (uint i = 0; i < MAX_JOBS2; ++i) {
    log_probs[i] = job_results[i].result;
  }
  if (MAX_JOBS2 == 0) {
    return 0.0;
  }
  return logsumexp(log_probs, MAX_JOBS2) - log(cdf_gauss_n) * 2 -
         log(cdf_gamma_n) * 2 - log(cdf_jbeta_n);
#endif
}
#endif


/* Compute Riemann sum to approximate the integral
 *
 * ∫ μ,σ  P[D,μ,σ]
 *
*/
double data_prob_1component_bySumming() {
#if defined(USE_THREADS)  // && (cdf_gauss_n > 4 * NUM_THREADS)
  return data_prob_1component_bySumming_parallel();
#else
#ifdef USE_LOG_PROB
  double *log_prob = malloc(cdf_gauss_n * cdf_gamma_n * sizeof(double));
  for (uint m = 0; m < cdf_gauss_n; ++m) {
    double mu = cdfInv_Gauss[m];
    for (uint s = 0; s < cdf_gamma_n; ++s) {
      double sigma = sigma_of_precision(cdfInv_gamma[s]);
      Gauss_params cur_params = {mu, sigma};
      log_prob[m * (uint) cdf_gamma_n + s] = prob_data_given_1Gauss(cur_params);
    }
  }
  double log_prob_total = logsumexp(log_prob, cdf_gauss_n * cdf_gamma_n);
  free(log_prob);
  return log_prob_total - log(cdf_gauss_n) - log(cdf_gamma_n);
#else
  double prob_total = 0.0;
  for (uint m = 0; m < cdf_gauss_n; ++m) {
    double mu = cdfInv_Gauss[m];
    for (uint s = 0; s < cdf_gamma_n; ++s) {
      double sigma = sigma_of_precision(cdfInv_gamma[s]);
      Gauss_params cur_params = {mu, sigma};
      prob_total += prob_data_given_1Gauss(cur_params);
    }
  }
  return prob_total / (double) (cdf_gauss_n * cdf_gamma_n);
#endif
#endif
}


/* Compute Riemann sum to approximate integral
 *
 * ∫ m,μ₁,σ₁,μ₂,σ₂  P[D,m,μ₁,σ₁,μ₂,σ₂]
 *
*/
double data_prob_2component_bySumming() {
#ifdef USE_THREADS
  return data_prob_2component_bySumming_parallel();
#else
#ifdef USE_LOG_PROB
  size_t idx = 0;
  double *log_prob = malloc(cdf_gauss_n * cdf_gauss_n * cdf_gamma_n *
                            cdf_gamma_n * cdf_jbeta_n * sizeof(double));
#else
  double prob_total = 0.0;
#endif

  for (uint m1 = 0; m1 < cdf_gauss_n; ++m1) {
    double mu1 = cdfInv_Gauss[m1];
    for (uint m2 = 0; m2 < cdf_gauss_n; ++m2) {
      double mu2 = cdfInv_Gauss[m2];
      for (uint s1 = 0; s1 < cdf_gamma_n; ++s1) {
        double sigma1 = sigma_of_precision(cdfInv_gamma[s1]);
        Gauss_params cur_params1 = {mu1, sigma1};
        for (uint s2 = 0; s2 < cdf_gamma_n; ++s2) {
          double sigma2 = sigma_of_precision(cdfInv_gamma[s2]);
          Gauss_params cur_params2 = {mu2, sigma2};
          for (uint mi = 0; mi < cdf_jbeta_n; ++mi) {
            double mixCof = cdfInv_JBeta[mi];
#ifdef USE_LOG_PROB
            log_prob[idx++] =
                prob_data_given_2Gauss(mixCof, cur_params1, cur_params2);
#else
            prob_total +=
                prob_data_given_2Gauss(mixCof, cur_params1, cur_params2);
#endif
          }
        }
      }
    }
  }
#ifdef USE_LOG_PROB
  double log_prob_total =
      logsumexp(log_prob, cdf_gauss_n * cdf_gauss_n * cdf_gamma_n *
                              cdf_gamma_n * cdf_jbeta_n);
  free(log_prob);
  return log_prob_total - log(cdf_gauss_n) * 2 - log(cdf_gamma_n) * 2 -
         log(cdf_jbeta_n);
#else
  return prob_total / (double) (cdf_gauss_n * cdf_gauss_n * cdf_gamma_n *
                                cdf_gamma_n * cdf_jbeta_n);
#endif
#endif
}


#ifdef USE_THREADS
void *thread_func_1component(void *arg) {
  ThreadArg *targ = (ThreadArg *) arg;
  init_rng(targ->thread_index);

#ifndef USE_LOG_PROB
  double sum = 0.0;
  for (uint i = targ->start; i < targ->end; ++i) {
    Gauss_params params = prior_Gauss_params_sample();
    sum += prob_data_given_1Gauss(params);
  }
  targ->result = sum;
#else
  size_t count = targ->end - targ->start;
  double *log_prob = malloc(count * sizeof(double));
  for (uint i = 0; i < count; ++i) {
    Gauss_params params = prior_Gauss_params_sample();
    log_prob[i] = prob_data_given_1Gauss(params);
  }
  if (count == 0) {
    targ->result = 0.0;
  } else {
    targ->result = logsumexp(log_prob, count);
  }
  free(log_prob);
#endif
  return NULL;
}


void *thread_func_2component(void *arg) {
  ThreadArg *targ = (ThreadArg *) arg;
  init_rng(targ->thread_index + 100);
#ifndef USE_LOG_PROB
  double sum = 0.0;
  for (uint i = targ->start; i < targ->end; ++i) {
    Gauss_mixture_params params = prior_Gauss_mixture_params_sample();
    sum += prob_data_given_2Gauss(params.mixCof, params.Gauss1, params.Gauss2);
  }
  targ->result = sum;
#else
  size_t count = targ->end - targ->start;
  double *log_prob = malloc(count * sizeof(double));
  for (uint i = 0; i < count; ++i) {
    Gauss_mixture_params params = prior_Gauss_mixture_params_sample();
    log_prob[i] =
        prob_data_given_2Gauss(params.mixCof, params.Gauss1, params.Gauss2);
  }
  if (count == 0) {
    targ->result = 0.0;
  } else {
    targ->result = logsumexp(log_prob, count);
  }
  free(log_prob);
#endif
  return NULL;
}
#endif


/*  Use sampling to estimate
 *  ∫ μ,σ  P[D,μ,σ]
 */
double data_prob_1component_bySampling() {
#ifndef USE_THREADS
  double prob_total = 0.0;

  for (uint iter = 0; iter < sampleRepeatNum; ++iter) {
    Gauss_params params = prior_Gauss_params_sample();
    prob_total += prob_data_given_1Gauss(params);
  }
  return prob_total / (double) sampleRepeatNum;
#else
  pthread_t threads[NUM_THREADS];
  ThreadArg args[NUM_THREADS];

  uint chunk = sampleRepeatNum / NUM_THREADS;
  uint remainder = sampleRepeatNum % NUM_THREADS;

  for (int i = 0; i < NUM_THREADS; ++i) {
    args[i].start = i * chunk;
    args[i].end =
        (i == NUM_THREADS - 1) ? (i + 1) * chunk + remainder : (i + 1) * chunk;
    args[i].result = 0.0;
    args[i].thread_index = i;
    pthread_create(&threads[i], NULL, thread_func_1component, &args[i]);
  }

#ifndef USE_LOG_PROB
  double total = 0.0;
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
    total += args[i].result;
  }
  return total / (double) sampleRepeatNum;
#else
  double log_probs[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
    log_probs[i] = args[i].result;
  }
  return logsumexp(log_probs, NUM_THREADS) - log(sampleRepeatNum);
#endif
#endif
}


/*  Use sampling to estimate
 *  ∫ m,μ₁,σ₁,μ₂,σ₂  P[D,m,μ₁,σ₁,μ₂,σ₂]
 */
double data_prob_2component_bySampling() {
#ifndef USE_THREADS
  double prob_total = 0.0;

  for (uint iter = 0; iter < sampleRepeatNum; ++iter) {
    Gauss_mixture_params params = prior_Gauss_mixture_params_sample();
    prob_total +=
        prob_data_given_2Gauss(params.mixCof, params.Gauss1, params.Gauss2);
  }
  return prob_total / (double) sampleRepeatNum;
#else
  pthread_t threads[NUM_THREADS];
  ThreadArg args[NUM_THREADS];

  uint chunk = sampleRepeatNum / NUM_THREADS;
  uint remainder = sampleRepeatNum % NUM_THREADS;

  for (int i = 0; i < NUM_THREADS; ++i) {
    args[i].start = i * chunk;
    args[i].end =
        (i == NUM_THREADS - 1) ? (i + 1) * chunk + remainder : (i + 1) * chunk;
    args[i].result = 0.0;
    args[i].thread_index = i;
    pthread_create(&threads[i], NULL, thread_func_2component, &args[i]);
  }
#ifndef USE_LOG_PROB
  double total = 0.0;
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
    total += args[i].result;
  }
  return total / (double) sampleRepeatNum;
#else
  double log_probs[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
    log_probs[i] = args[i].result;
  }
  return logsumexp(log_probs, NUM_THREADS) - log(sampleRepeatNum);
#endif
#endif
}

char savename[100];


void parse_args(int argc, char *argv[]) {
  if (argc >= 2) data_n = atoi(argv[1]);
  if (argc >= 3) dataset_n = atoi(argv[2]);
  if (argc >= 4) cdf_gauss_n = atoi(argv[3]);
  if (argc >= 5) cdf_gamma_n = atoi(argv[4]);
  if (argc >= 6) cdf_jbeta_n = atoi(argv[5]);
  if (argc >= 7) {
    snprintf(savename, sizeof(savename), "%s", argv[6]);
  } else {
    snprintf(savename, sizeof(savename), "Gaussian_poolOrNot_result.txt");
  }
  if (argc >= 8) mu_prior_params.mu = atof(argv[7]);
  if (argc >= 9) mu_prior_params.sigma = atof(argv[8]);
  if (argc >= 10) sigma_prior_param_a = atof(argv[9]);
  if (argc >= 11) sigma_prior_param_b = atof(argv[10]);
  if (argc >= 12) sampleRepeatNum = atoi(argv[11]);
}


int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  GSLfun_setup();
  double prob_data1_bySampling, prob_data2_bySampling;
  double prob_data1_bySumming, prob_data2_bySumming;

  cdfInv_precompute();

  uint model1_sampling_favors1 = 0;
  uint model1_summing__favors1 = 0;
  uint model2_sampling_favors1 = 0;
  uint model2_summing__favors1 = 0;

#ifndef PRINT_RESULTS_ONLY
  printf("Starting computation for %d datasets each. ...\n", dataset_n);
#endif

  Gauss_params model_params_list[dataset_n];
  Gauss_mixture_params model_params_list2[dataset_n];
  for (uint i = 0; i < dataset_n; ++i) {
    model_params_list[i] = prior_Gauss_params_sample();
    model_params_list2[i] = prior_Gauss_mixture_params_sample();
  }

  data = malloc(data_n * sizeof(double));
  double data1s[dataset_n][data_n];
  double data2s[dataset_n][data_n];

#ifdef USE_THREADS
  job_results = malloc(MAX_JOBS2 * sizeof(JobResult));
#endif

  for (
      size_t i = 0; i < data_n;
      ++i) {  // loop the data first to make sure in different data_n, they contain the same data
    for (size_t j = 0; j < dataset_n; ++j) {  // loop the datasets
      data1s[j][i] = GSLfun_ran_gaussian(model_params_list[j]);
      data2s[j][i] =
          GSLfun_ran_gaussian(gsl_ran_flat01() < model_params_list2[j].mixCof
                                  ? model_params_list2[j].Gauss1
                                  : model_params_list2[j].Gauss2);
    }
  }

#ifndef PRINT_RESULTS_ONLY
  printf("\nData generated with one component\n");
#endif
  for (uint iter = 0; iter < dataset_n; ++iter) {
#ifndef PRINT_RESULTS_ONLY
    Gauss_params model_params = model_params_list[iter];
    printf("generating data with: (μ,σ) =  (%4.2f,%4.2f)\n", model_params.mu,
           model_params.sigma);
#endif
    memcpy(data, data1s[iter],
           sizeof(double) * data_n);  // copy the data from the array
    init_prob_data_1gauss();  // initialize the probabilities for the data

#ifndef PRINT_RESULTS_ONLY
    printf("Data maximum likelihood under one component model= %g\n",
           data_Gauss1_maxLikelihood());
#endif

    prob_data1_bySampling = data_prob_1component_bySampling();
    prob_data2_bySampling = data_prob_2component_bySampling();
    prob_data1_bySumming = data_prob_1component_bySumming();
    prob_data2_bySumming = data_prob_2component_bySumming();
#ifndef PRINT_RESULTS_ONLY
    printf("Integrals by sampling= (%g,%g)  by summing: (%g,%g)\n\n",
           prob_data1_bySampling, prob_data2_bySampling, prob_data1_bySumming,
           prob_data2_bySumming);
#endif
    if (prob_data1_bySampling > prob_data2_bySampling)
      ++model1_sampling_favors1;
    if (prob_data1_bySumming > prob_data2_bySumming) ++model1_summing__favors1;
  }

#ifndef PRINT_RESULTS_ONLY
  printf("\nData generated with two components\n");
#endif
  for (uint iter = 0; iter < dataset_n; ++iter) {
#ifndef PRINT_RESULTS_ONLY
    Gauss_mixture_params model_params = model_params_list2[iter];
    printf(
        "generating data with:  m; (μ1,σ1); (μ2,σ2) =  %5.3f; (%4.2f,%4.2f); "
        "(%4.2f,%4.2f)\n",
        model_params.mixCof, model_params.Gauss1.mu, model_params.Gauss1.sigma,
        model_params.Gauss2.mu, model_params.Gauss2.sigma);
#endif
    memcpy(data, data2s[iter],
           sizeof(double) * data_n);  // copy the data from the array
    init_prob_data_1gauss();  // initialize the probabilities for the data
#ifndef PRINT_RESULTS_ONLY
    printf("Data maximum likelihood under one component model= %g\n",
           data_Gauss1_maxLikelihood());
#endif

    prob_data1_bySampling = data_prob_1component_bySampling();
    prob_data2_bySampling = data_prob_2component_bySampling();
    prob_data1_bySumming = data_prob_1component_bySumming();
    prob_data2_bySumming = data_prob_2component_bySumming();
#ifndef PRINT_RESULTS_ONLY
    printf("Integrals by sampling= (%g,%g)  by summing: (%g,%g)\n\n",
           prob_data1_bySampling, prob_data2_bySampling, prob_data1_bySumming,
           prob_data2_bySumming);
#endif
    if (prob_data1_bySampling > prob_data2_bySampling)
      ++model2_sampling_favors1;
    if (prob_data1_bySumming > prob_data2_bySumming) ++model2_summing__favors1;
  }

  printf("(%d, %d, %d, %d)\n", model1_sampling_favors1,
         (dataset_n - model2_sampling_favors1), model1_summing__favors1,
         (dataset_n - model2_summing__favors1));

  free(data);
  free(cdfInv_Gauss);
  free(cdfInv_gamma);
  free(cdfInv_JBeta);

#ifdef USE_THREADS
  free(job_results);
  pthread_key_delete(rng_key);
  pthread_mutex_destroy(&job_mutex);
#endif
}
