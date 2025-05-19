#include <pthread.h>

#include "GSLfun.h"

#define USE_THREADS

#ifdef USE_THREADS
#define NUM_THREADS 16

typedef struct {
  uint start, end;
  double result;
  int thread_index;
} ThreadArg;

#endif
