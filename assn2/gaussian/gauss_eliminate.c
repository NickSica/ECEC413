/* Gaussian elimination code.
 *
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student name(s): Nicholas Sica and Cameron Calv
 * Date: 5/1/2020
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#include "gauss_eliminate.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }
    
    int matrix_size = atoi(argv[1]);

    Matrix A;			                                    /* Input matrix */
    Matrix U_reference;		                                    /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                            /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    //fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    //gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    //gettimeofday(&stop, NULL);
    //fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    gettimeofday(&start, NULL);
    gauss_eliminate_using_pthreads(U_mt);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    
    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* Perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U)
{
    pthread_barrierattr_t barrier_attr;
    pthread_barrierattr_init(&barrier_attr);
    pthread_t *tid = malloc(NUM_THREADS * sizeof(pthread_t));
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    for(int i = 0; i < U.num_rows; i++)
    {
	int chunk_rows = U.num_rows / NUM_THREADS;
        int start_row = i * U.num_columns + i;
	int end_row = (i + 1) * U.num_columns;
	float piv_element = U.elements[start_row];
	U.elements[start_row] = 1;
	start_row++;

	pthread_barrier_t barrier;
	pthread_barrier_init(&barrier, &barrier_attr, NUM_THREADS + 1);
	thread_data_t *thread_data = malloc(NUM_THREADS * sizeof(thread_data_t));
	for(int j = 0; j < NUM_THREADS; j++)
	{
	    thread_data[j].div_start = start_row + j;
	    thread_data[j].div_end = end_row;
	    thread_data[j].chunk_rows = chunk_rows;
	    thread_data[j].elim_start = i + 1 + (j * chunk_rows);
	    thread_data[j].num_iter = i;
	    thread_data[j].tid = j;
	    thread_data[j].piv_element = piv_element;
	    thread_data[j].matrix = &U;
	    thread_data[j].barrier = &barrier;
	}
	
	for(int j = 0; j < NUM_THREADS; j++)
	    pthread_create(&tid[j], &thread_attr, gauss_reduce, (void *)&thread_data[j]);

        pthread_barrier_wait(&barrier);

	for(int j = 0; j < NUM_THREADS; j++)
	    pthread_join(tid[j], NULL);
	free(thread_data);
    }

    free(tid);
}

void *gauss_reduce(void *args)
{
    // Divide row i with the pivot element
    thread_data_t *thread_data = (thread_data_t *)args;
    Matrix *matrix = thread_data->matrix;
    for(int i = thread_data->div_start; i < thread_data->div_end; i += NUM_THREADS)
	matrix->elements[i] = matrix->elements[i] / thread_data->piv_element;

    pthread_barrier_wait(thread_data->barrier);
   
    // Eliminate rows (i + 1) to (n - 1)
    int num_elements = matrix->num_rows;
    int elim_end = thread_data->num_iter + 1 + ((thread_data->tid + 1) * thread_data->chunk_rows);
    if (elim_end > num_elements)
	elim_end = num_elements;
	 
    for (int i = thread_data->elim_start; i < elim_end; i++)
    {
	for (int j = (thread_data->num_iter + 1); j < num_elements; j++)
	    matrix->elements[num_elements * i + j] = matrix->elements[num_elements * i + j] - (matrix->elements[num_elements * i + thread_data->num_iter] * matrix->elements[num_elements * thread_data->num_iter + j]);
            
	matrix->elements[num_elements * i + thread_data->num_iter] = 0;
    }

    pthread_exit(NULL);
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
	    return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}
