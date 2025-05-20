#include "GSLfun.h"

#include "thread_setup.h"

#ifdef USE_THREADS
int thread_idx = 0;
static pthread_key_t rng_key;
static pthread_once_t rng_key_once = PTHREAD_ONCE_INIT;

static void make_key() {
  pthread_key_create(&rng_key, (void (*)(void *)) gsl_rng_free);
}

static void init_rng() {
  gsl_rng_env_setup();
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, DEFAULT_RNG_SEED);
  pthread_setspecific(rng_key, r);
}
#else
static gsl_rng *gslRNG = NULL;
#endif

static gsl_rng *get_rng() {
#ifdef USE_THREADS
  pthread_once(&rng_key_once, make_key);
  gsl_rng *r = pthread_getspecific(rng_key);
  if (!r) {
    init_rng();  // Initialize the RNG if not already done
    r = pthread_getspecific(rng_key);
  }
  return r;
#else
  return gslRNG;
#endif
}


void Gauss_params_print(const Gauss_params params) {
  printf("(%+4.2f,%4.2f)", params.mu, params.sigma);
}

void GSLfun_setup() {
#ifndef USE_THREADS
  if (!getenv("GSL_RNG_SEED")) printf("Using default random seed\n");
  gsl_rng_env_setup();
  gslRNG = gsl_rng_alloc(gsl_rng_mt19937);
#endif
}


double GSLfun_ran_beta(double a, double b) {
  return gsl_ran_beta(get_rng(), a, b);
}

double GSLfun_ran_beta_Jeffreys() { return gsl_ran_beta(get_rng(), 0.5, 0.5); }

uint GSLfun_ran_binomial(double p, uint n) {
  return gsl_ran_binomial(get_rng(), p, n);
}

double GSLfun_ran_gamma(double a, double theta) {
  return gsl_ran_gamma(get_rng(), a, theta);
}

double GSLfun_ran_gaussian(const Gauss_params params) {
  return params.mu + gsl_ran_gaussian(get_rng(), params.sigma);
}

double GSLfun_ran_gaussian_pdf(double x, const Gauss_params params) {
  return gsl_ran_gaussian_pdf(x - params.mu, params.sigma);
}

double gsl_ran_flat01() { return gsl_ran_flat(get_rng(), 0.0, 1.0); }

double sigma_of_precision(double precision) { return sqrt(1.0 / precision); }
