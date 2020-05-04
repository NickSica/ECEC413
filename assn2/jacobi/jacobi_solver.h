#ifndef _JACOBI_SOLVER_H_
#define _JACOBI_SOLVER_H_

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#define THRESHOLD 1e-5      /* Threshold for convergence */
#define MIN_NUMBER 2        /* Min number in the A and b matrices */
#define MAX_NUMBER 10       /* Max number in the A and b matrices */
#define NUM_THREADS 8

/* Matrix structure declaration */
typedef struct matrix_s {
    unsigned int num_columns;   /* Matrix width */
    unsigned int num_rows;      /* Matrix height */ 
    float *elements;
} matrix_t;

typedef struct init_data_s {
    int start;
    int stride_size;
    int num_elements;
    matrix_t *x;
    const matrix_t *B;
} init_data_t;

typedef struct thread_data_s {
    int tid;
    int start_row;
    int chunk_rows;
    unsigned int num_columns;
    unsigned int num_rows;
    const matrix_t *A;
    const matrix_t *B;
    matrix_t *x;
    matrix_t *new_x;
    double *ssd;    
    pthread_mutex_t *ssd_lock;
    pthread_barrier_t *barrier;
} thread_data_t;

/* Function prototypes */
matrix_t allocate_matrix (int, int, int);
extern void compute_gold(const matrix_t, matrix_t, const matrix_t, int);
extern void display_jacobi_solution(const matrix_t, const matrix_t, const matrix_t);
int check_if_diagonal_dominant(const matrix_t);
matrix_t create_diagonally_dominant_matrix(int, int);
void compute_using_pthreads(const matrix_t, matrix_t, const matrix_t);
void *jacobi_init(void *args);
void *compute_and_check(void *args);
void print_matrix(const matrix_t);
float get_random_number(int, int);

#endif /* _JACOBI_SOLVER_H_ */

