#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 1

/* Matrix structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */ 
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

// Division step structure declaration
typedef struct thread_data_s {
    int chunk_rows;
    int tid;
    int div_start;
    int div_end;
    int elim_start;
    int num_iter;
    float piv_element;
    Matrix *matrix;
    pthread_barrier_t *barrier;
} thread_data_t;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);
void *gauss_reduce(void *args);

#endif /* _MATRIXMUL_H_ */
