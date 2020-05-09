/* Gaussian elimination code.
 *
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student name(s): Nicholas Sica and Cameron Calv
 * Date: 5/8/2020
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -O3 -Wall -lm
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
    fprintf(stderr, "\nPerforming gaussian elimination using openmp\n");
    gettimeofday(&start, NULL);
    gauss_eliminate_using_openmp(U_mt);
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


/* Perform gaussian elimination using openmp */
void gauss_eliminate_using_openmp(Matrix U)
{
    int div_start, elim_start;
    int tid;
    const int NUM_THREADS = omp_get_num_threads();
    int chunk_rows = U.num_rows / NUM_THREADS;
    for(int i = 0; i < U.num_rows; i++)
    {
        int start_row = i * U.num_columns + i;
	int end_row = (i + 1) * U.num_columns;
	float piv_element = U.elements[start_row];
	U.elements[start_row] = 1;
	start_row++;
		
#pragma omp parallel default(none) private(tid, div_start, elim_start) shared(U, NUM_THREADS, i, chunk_rows, start_row, end_row, piv_element)
	{
	    tid = omp_get_thread_num();
	    div_start = start_row + tid;
	    elim_start = i + 1 + (tid * chunk_rows);
		
	    // Divide row i with the pivot element
//#pragma omp for nowait
	    for(int i = div_start; i < end_row; i += NUM_THREADS)
		U.elements[i] = U.elements[i] / piv_element;
	    
#pragma omp barrier
	    
	    // Eliminate rows (i + 1) to (n - 1)
	    int num_elements = U.num_rows;
	    int elim_end = i + 1 + ((tid + 1) * chunk_rows);
	    if (elim_end > num_elements)
		elim_end = num_elements;

//#pragma omp for nowait
	    for (int i = elim_start; i < elim_end; i++)
	    {
		for (int j = (i + 1); j < num_elements; j++)
		    U.elements[num_elements * i + j] = U.elements[num_elements * i + j] - (U.elements[num_elements * i + i] * U.elements[num_elements * i + j]);
            
		U.elements[num_elements * i + i] = 0;
	    }
	    
#pragma omp barrier
	}
    }
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
