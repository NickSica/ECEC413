#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <time.h>
#include <sys/time.h>

#include <omp.h>

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define ITER_OUTPUT 1

/* Matrix structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */ 
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_openmp(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

#endif /* _MATRIXMUL_H_ */
