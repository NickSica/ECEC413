/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 21, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */
/* #define DEBUG */

int main(int argc, char **argv) 
{
    if (argc > 1)
    {
	printf("This program accepts no arguments\n");
	exit(EXIT_FAILURE);
    }

    matrix_t  A;                    /* N x N constant matrix */
    matrix_t  B;                    /* N x 1 b matrix */
    matrix_t reference_x;           /* Reference solution */ 
    matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */
    
    /* Initialize the random number generator */
    srand(time(NULL));
    
    /* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
    A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
    if (A.elements == NULL)
    {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
    }
    
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
    reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif
    
    /* Compute Jacobi solution on CPU */
    printf("\nPerforming Jacobi iteration on the CPU\n");
    int num_iters = compute_gold(A, reference_x, B);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
    
    /* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\n\nPerforming Naive Jacobi iteration on device\n");
    compute_on_device_naive(A, gpu_naive_solution_x, B, num_iters);
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */

    printf("\n\nPerforming Optimized Jacobi iteration on device\n");
    compute_on_device_opt(A, gpu_opt_solution_x, B);
    display_jacobi_solution(A, gpu_opt_solution_x, B); 
    
    free(A.elements); 
    free(B.elements); 
    free(reference_x.elements); 
    free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


/* Perform Jacobi calculation on device using naive kernel*/
void compute_on_device_naive(const matrix_t A, matrix_t gpu_sol_x, const matrix_t B, int num_iters)
{
    matrix_t d_A = allocate_matrix_on_device(A);
    matrix_t d_B = allocate_matrix_on_device(B);
    matrix_t d_gpu_sol_x = allocate_matrix_on_device(gpu_sol_x);
    double *d_ssd;
    cudaMalloc((void **)&d_ssd, sizeof(double));

    /* Initialize current naive jacobi solution */
    for (int i = 0; i < A.num_rows; i++)
        gpu_sol_x.elements[i] = B.elements[i];

    copy_matrix_to_device(d_A, A);
    copy_matrix_to_device(d_B, B);
    copy_matrix_to_device(d_gpu_sol_x, gpu_sol_x);
    check_CUDA_error("Kernel Setup Failure");
    int num_threads = min((int)THREAD_BLOCK_SIZE, (int)MATRIX_SIZE);
    
    dim3 threads(num_threads, 1, 1);
    dim3 grid(MATRIX_SIZE / threads.x, 1);
    
    /* Perform Naive Jacobi iteration */
    unsigned int done = 0;
    double mse, ssd;
    unsigned int num_iter = 0;
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    while(!done)
    {
	cudaMemset(d_ssd, 0, sizeof(double));
	check_CUDA_error("CUDA Memset Failure");

	jacobi_iteration_kernel_naive<<<grid, threads>>>(d_A.elements, d_B.elements, d_A.num_columns, d_A.num_rows, d_gpu_sol_x.elements, d_ssd);
	cudaDeviceSynchronize();
	check_CUDA_error("Kernel Launch Failure");

        cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);
	check_CUDA_error("CUDA Memcpy Failure");
       
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
#ifdef DISPLAY_ITERS	
	printf("Iteration: %d. MSE = %f\n", num_iter, mse); 
#endif
	
        if (mse <= THRESHOLD)
            done = 1;
    }
    gettimeofday(&stop, NULL);

    printf("\nNaive convergence achieved after %d iterations \n", num_iter);
    printf("Naive execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
    copy_matrix_from_device(gpu_sol_x, d_gpu_sol_x);

    cudaFree(d_gpu_sol_x.elements);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_ssd);

    return;
}

void compute_on_device_opt(const matrix_t A, matrix_t gpu_sol_x, const matrix_t B)
{
    // Setup A in column major form
    matrix_t d_A = allocate_matrix_on_device(A);
    matrix_t col_major_A = allocate_matrix_on_host(MATRIX_SIZE, MATRIX_SIZE, 0);
    matrix_t d_col_major_A = allocate_matrix_on_device(col_major_A);
    copy_matrix_to_device(d_A, A);
    copy_matrix_to_device(d_col_major_A, col_major_A);

    int num_threads = min((int)THREAD_BLOCK_SIZE / 4, (int)MATRIX_SIZE);
    dim3 row_to_col_threads(num_threads, num_threads, 1);
    dim3 row_to_col_grid(MATRIX_SIZE / row_to_col_threads.x, MATRIX_SIZE / row_to_col_threads.y, 1);
    row_to_col_major_kernel<<<row_to_col_grid, row_to_col_threads>>>(d_A.elements, d_A.num_columns, d_A.num_rows, d_col_major_A.elements);
    cudaDeviceSynchronize();
    check_CUDA_error("Row to Column Major Kernel Launch Failure");

    // Setup device variables
    matrix_t d_B = allocate_matrix_on_device(B);
    matrix_t d_gpu_sol_x = allocate_matrix_on_device(gpu_sol_x);
    double *d_ssd;
    cudaMalloc((void **)&d_ssd, sizeof(double));
        
    /* Initialize current optimized jacobi solution */
    for (int i = 0; i < B.num_rows; i++)
        gpu_sol_x.elements[i] = B.elements[i];

    copy_matrix_to_device(d_B, B);
    copy_matrix_to_device(d_gpu_sol_x, gpu_sol_x);
    check_CUDA_error("Kernel Setup Failure");

    num_threads = min((int)THREAD_BLOCK_SIZE, (int)MATRIX_SIZE);
    dim3 threads(num_threads, 1, 1);
    dim3 grid(MATRIX_SIZE / threads.x, 1);
    
    /* Perform Jacobi iteration */
    unsigned int done = 0;
    double mse, ssd;
    unsigned int num_iter = 0;
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    while (!done)
    {
	cudaMemset(d_ssd, 0, sizeof(double));
	check_CUDA_error("CUDA Memset Failure");
	
	jacobi_iteration_kernel_optimized<<<grid, threads>>>(d_col_major_A.elements, d_B.elements, d_col_major_A.num_columns, d_col_major_A.num_rows, d_gpu_sol_x.elements, d_ssd);
	cudaDeviceSynchronize();
	check_CUDA_error("Optimized Kernel Launch Failure");

	cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);
	check_CUDA_error("CUDA Memcpy Failure");
	
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
#ifdef DISPLAY_ITERS
        printf("Iteration: %d. MSE = %f\n", num_iter, mse);
#endif
        
        if (mse <= THRESHOLD)
            done = 1;
    }
    gettimeofday(&stop, NULL);
    
    printf("\nOptimized convergence achieved after %d iterations \n", num_iter);
    printf("Optimized execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
    copy_matrix_from_device(gpu_sol_x, d_gpu_sol_x);
    
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_gpu_sol_x.elements);
    cudaFree(d_ssd);
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
    M.elements = (float *)malloc(size * sizeof(float));
    for (unsigned int i = 0; i < size; i++) {
	if (init == 0) 
            M.elements[i] = 0; 
	else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
    
    return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_rows + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

