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
#define NUM_THREADS 4

/* Matrix structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */ 
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

// Division step structure declaration
typedef struct divide_data_s {
    int size;
    int start;
    int end;
    float piv_element;
    float *elements;
    pthread_barrier_t *barrier;
} divide_data_t;

// Elimination step structure declaration
typedef struct elim_data_s {
    int tid;
    int chunk_size;
    int start;
    int num_iter;
    float piv_element;
    //    unsigned int num_columns;
    //
    // unsigned int num_rows;
//    float *elements;
    Matrix *matrix;
    pthread_barrier_t *barrier;
} elim_data_t;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);
void *divide(void *args);
void *eliminate(void *args);

#endif /* _MATRIXMUL_H_ */

