/* Code for the Jacobi method of solving a system of linear equations
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Student name(s): Nicholas Sica and Cameron Calv
 * Date: 5/1/2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#include "jacobi_solver.h"
#include <pthread.h>

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
    if (argc < 2) {
	fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
	exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);

    matrix_t  A;                    /* N x N constant matrix */
    matrix_t  B;                    /* N x 1 b matrix */
    matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
    srand(time(NULL));
    A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
    if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
    }
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
    reference_x = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif

	 struct timeval start, stop;

    /* Compute Jacobi solution using reference code */
    fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
    /* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
    gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    
    free(A.elements); 
    free(B.elements); 
    free(reference_x.elements); 
    free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B)
{
    int max_iter = 100000;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      

    /* Initialize current jacobi solution. */
    pthread_t *tid = malloc(NUM_THREADS * sizeof(pthread_t));
    init_data_t *init_data = malloc(NUM_THREADS * sizeof(init_data_t));
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    for(int i = 0; i < NUM_THREADS; i++)
    {
	init_data[i].start = i;
	init_data[i].num_elements = num_rows;
	init_data[i].stride_size = NUM_THREADS;
	init_data[i].x = &mt_sol_x;
	init_data[i].B = &B;	
    }

    for(int i = 0; i < NUM_THREADS; i++)
	pthread_create(&tid[i], &thread_attr, jacobi_init, (void *)&init_data[i]);

    for(int i = 0; i < NUM_THREADS; i++)
	pthread_join(tid[i], NULL);
    
    /* Perform Jacobi iteration. */
    int chunk_rows = (int)floor((float)num_rows / (float)NUM_THREADS);
    int done = 0;
    double ssd = 0.0;
    double mse;
    int num_iter = 0;

    pthread_barrierattr_t barrier_attr;
    pthread_barrierattr_init(&barrier_attr);	
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, &barrier_attr, NUM_THREADS + 1);

    pthread_mutex_t ssd_lock;
    pthread_mutex_init(&ssd_lock, NULL);
    
    thread_data_t *thread_data = malloc(NUM_THREADS * sizeof(thread_data_t));
    for(int i = 0; i < NUM_THREADS; i++)
    {
	thread_data[i].tid = i;
	thread_data[i].start_row = i * chunk_rows;
	thread_data[i].chunk_rows = chunk_rows;
	thread_data[i].num_columns = num_cols;
	thread_data[i].num_rows = num_rows;
	thread_data[i].A = &A;
	thread_data[i].B = &B;
	thread_data[i].x = &mt_sol_x;
	thread_data[i].new_x = &new_x;
	thread_data[i].ssd = &ssd;
	thread_data[i].ssd_lock = &ssd_lock;
	thread_data[i].barrier = &barrier;
    }

    while(!done)
    {
	ssd = 0.0;
       	for(int i = 0; i < NUM_THREADS; i++)
	    pthread_create(&tid[i], &thread_attr, compute_and_check, (void *)&thread_data[i]);

	pthread_barrier_wait(&barrier);

	for(int i = 0; i < NUM_THREADS; i++)
	    pthread_join(tid[i], NULL);

	num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        
        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
    }

    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");

    free(new_x.elements);
    free(tid);
    free(thread_data);
}

void *jacobi_init(void *args)
{
    init_data_t *thread_data = (init_data_t *)args;
    for(int i = thread_data->start; i < thread_data->num_elements; i += thread_data->stride_size)
	thread_data->x->elements[i] = thread_data->B->elements[i];
    pthread_exit(NULL);
}

void *compute_and_check(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
    const matrix_t *A = thread_data->A;
    const matrix_t *B = thread_data->B;
    matrix_t *x = thread_data->x;
    matrix_t *new_x = thread_data->new_x;
    int end_row = thread_data->start_row + thread_data->chunk_rows;
    if(thread_data->tid >= NUM_THREADS - 1)
	end_row = thread_data->num_rows;
    
    for(int i = thread_data->start_row; i < end_row; i++)
    {
	double sum = 0.0;
	for(int j = 0; j < thread_data->num_columns; j++)
	    if(i != j)
		sum += A->elements[i * thread_data->num_columns + j] * x->elements[j];

        /* Update values for the unkowns for the current row. */
	new_x->elements[i] = (B->elements[i] - sum) / A->elements[i * thread_data->num_columns + i];
    }
    
    pthread_barrier_wait(thread_data->barrier);
    
    /* Check for convergence and update the unknowns. */
    double ssd = 0.0;
    for(int i = thread_data->start_row; i < end_row; i++)
    {
	ssd += (new_x->elements[i] - x->elements[i]) * (new_x->elements[i] - x->elements[i]);
	x->elements[i] = new_x->elements[i];
    }
    pthread_mutex_lock(thread_data->ssd_lock);
    *(thread_data->ssd) += ssd;
    pthread_mutex_unlock(thread_data->ssd_lock);
    pthread_exit(NULL);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
    for (i = 0; i < size; i++)
    {
	if (init == 0) 
	    M.elements[i] = 0; 
	else
	    M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
    
    return M;
}	

/* Print
 matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



