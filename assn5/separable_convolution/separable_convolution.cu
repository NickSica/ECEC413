/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

float *allocate_on_device(const float *, int, int);
void copy_to_device(float *, const float *, int, int);
void copy_from_device(float *, const float *, int, int);
void compare_results(float *, float *, int);
extern "C" void compute_gold(float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);
void check_for_error(char *);
void print_kernel(float *, int);
void print_matrix(float *, int, int);

/* Width of convolution kernel */
#define HALF_WIDTH 8
#define COEFF 10

#define THREAD_BLOCK_SIZE 32

/* Uncomment line below to spit out debug information */
// #define DEBUG

/* Include device code */
#include "separable_convolution_kernel.cu"

/*  Computes the convolution on the device using the naive kernel.*/
void compute_on_device_naive(float *gpu_result, float *matrix_c, \
			     float *kernel, int num_cols,	 \
			     int num_rows, int half_width)
{
    float *gpu_row_result = (float *)malloc(sizeof(float) * num_rows * num_cols);
    
    float *d_gpu_col_result = allocate_on_device(gpu_result, num_rows, num_cols);
    float *d_gpu_row_result = allocate_on_device(gpu_result, num_rows, num_cols);
    float *d_kernel = allocate_on_device(kernel, num_rows, num_cols);
    float *d_matrix_c = allocate_on_device(matrix_c, num_rows, num_cols);

    dim3 threads(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1);
    int num_thread_blocks_x = ceil((float)num_cols / (float)threads.x);
    int num_thread_blocks_y = ceil((float)num_rows / (float)threads.y);
    dim3 grid(num_thread_blocks_x, num_thread_blocks_y, 1);

    copy_to_device(d_matrix_c, matrix_c, num_rows, num_cols);
    copy_to_device(d_kernel, kernel, num_rows, num_cols);

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    convolve_rows_kernel_naive<<<grid, threads>>>(d_gpu_row_result, d_matrix_c, d_kernel, num_cols, num_rows);
    cudaDeviceSynchronize();
    check_for_error("Error launching naive row convolution.");

    convolve_columns_kernel_naive<<<grid, threads>>>(d_gpu_col_result, d_gpu_row_result, d_kernel, num_cols, num_rows);
    cudaDeviceSynchronize();
    check_for_error("Error launching naive column convolution.");
    
    gettimeofday(&stop, NULL);
    printf("Naive execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
    
    copy_from_device(gpu_result, d_gpu_col_result, num_rows, num_cols);
    
    free(gpu_row_result);
    cudaFree(d_gpu_row_result);
    cudaFree(d_gpu_col_result);
    cudaFree(d_kernel);
    cudaFree(d_matrix_c);
    return;
}

/*  Computes the convolution on the device using the optimized kernel.*/
void compute_on_device_optimized(float *gpu_result, float *matrix_c, \
				 float *kernel, int num_cols,	     \
				 int num_rows, int half_width)
{
    int kernel_width = half_width * 2 + 1;
    float *gpu_row_result = (float *)malloc(sizeof(float) * num_rows * num_cols);
    
    float *d_gpu_col_result = allocate_on_device(gpu_result, num_rows, num_cols);
    float *d_gpu_row_result = allocate_on_device(gpu_result, num_rows, num_cols);
    float *d_matrix_c = allocate_on_device(matrix_c, num_rows, num_cols);

    dim3 threads(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1);
    int num_thread_blocks_x = ceil((float)num_cols / (float)threads.x);
    int num_thread_blocks_y = ceil((float)num_rows / (float)threads.y);
    dim3 grid(num_thread_blocks_x, num_thread_blocks_y, 1);
    
    copy_to_device(d_matrix_c, matrix_c, num_rows, num_cols);
    cudaMemcpyToSymbol(kernel_c, kernel, kernel_width * sizeof(float));
    check_for_error("Error setting up optimized convolution.");
    
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    convolve_rows_kernel_optimized<<<grid, threads>>>(d_gpu_row_result, d_matrix_c, num_cols, num_rows);
    cudaDeviceSynchronize();
    check_for_error("Error launching optimized row convolution.");

    convolve_columns_kernel_optimized<<<grid, threads>>>(d_gpu_col_result, d_gpu_row_result, num_cols, num_rows);
    cudaDeviceSynchronize();
    check_for_error("Error launching optimized column convolution.");

    gettimeofday(&stop, NULL);
    printf("Optimized execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    copy_from_device(gpu_result, d_gpu_col_result, num_rows, num_cols);

    free(gpu_row_result);
    cudaFree(d_gpu_row_result);
    cudaFree(d_gpu_col_result);
    cudaFree(d_matrix_c);
    return;
}

/* Allocate matrix on the device of same size as M */
float *allocate_on_device(const float *M, int num_rows, int num_columns)
{
    float *Mdevice;
    int size = num_rows * num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice, size);
    return Mdevice;
}

/* Copy matrix to device */
void copy_to_device(float *Mdevice, const float *Mhost, int num_rows, int num_columns)
{
    int size = num_rows * num_columns * sizeof(float);
    cudaMemcpy(Mdevice, Mhost, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_from_device(float *Mhost, const float *Mdevice, int num_rows, int num_columns)
{
    int size = num_rows * num_columns * sizeof(float);
    cudaMemcpy(Mhost, Mdevice, size, cudaMemcpyDeviceToHost);
    return;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s num-rows num-columns\n", argv[0]);
        printf("num-rows: height of the matrix\n");
        printf("num-columns: width of the matrix\n");
        exit(EXIT_FAILURE);
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);

    /* Create input matrix */
    int num_elements = num_rows * num_cols;
    printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
    float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
    float *matrix_c = (float *)malloc(sizeof(float) * num_elements);
	
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++) {
        matrix_a[i] = rand()/(float)RAND_MAX;			 
        matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
    }
	 
	/* Create Gaussian kernel */	  
    float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
#ifdef DEBUG
    print_kernel(gaussian_kernel, HALF_WIDTH); 
#endif
	  
    /* Convolve matrix along rows and columns. 
       The result is stored in matrix_a, thereby overwriting the 
       original contents of matrix_a.		
     */
    printf("\nConvolving the matrix on the CPU\n");	  
    compute_gold(matrix_a, gaussian_kernel, num_cols,\
                  num_rows, HALF_WIDTH);
#ifdef DEBUG	 
    print_matrix(matrix_a, num_cols, num_rows);
#endif
  
    float *gpu_result_naive = (float *)malloc(sizeof(float) * num_elements);
    float *gpu_result_opt = (float *)malloc(sizeof(float) * num_elements);
    
    /* FIXME: Edit this function to complete the functionality on the GPU.
       The input matrix is matrix_c and the result must be stored in 
       gpu_result.
     */
    printf("\nConvolving matrix on the GPU (Naive)\n");
    compute_on_device_naive(gpu_result_naive, matrix_c, gaussian_kernel, num_cols,\
                            num_rows, HALF_WIDTH);
    
    printf("\nComparing CPU and Naive GPU results\n");
    compare_results(matrix_a, gpu_result_naive, num_elements);

    printf("\nConvolving matrix on the GPU (Optimized)\n");
    compute_on_device_optimized(gpu_result_opt, matrix_c, gaussian_kernel, num_cols,\
                                num_rows, HALF_WIDTH);

    printf("\nComparing CPU and Optimized GPU results\n");
    compare_results(matrix_a, gpu_result_opt, num_elements);

    free(matrix_a);
    free(matrix_c);
    free(gpu_result_naive);
    free(gpu_result_opt);
    free(gaussian_kernel);

    exit(EXIT_SUCCESS);
}

void compare_results(float *expected, float *actual, int num_elements)
{
    float sum_delta = 0, sum_ref = 0;
    for (int i = 0; i < num_elements; i++) {
        sum_delta += fabsf(expected[i] - actual[i]);
        sum_ref   += fabsf(expected[i]);
    }
        
    float L1norm = sum_delta / sum_ref;
    float eps = 1e-6;
    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

}

/* Check for errors reported by the CUDA run time */
void check_for_error(char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
	printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    
    return;
} 

/* Print convolution kernel */
void print_kernel(float *kernel, int half_width)
{
    int i, j = 0;
    for (i = -half_width; i <= half_width; i++)
    {
        printf("%0.2f ", kernel[j]);
        j++;
    }

    printf("\n");
    return;
}

/* Print matrix */
void print_matrix(float *matrix, int num_cols, int num_rows)
{
    int i,  j;
    float element;
    for (i = 0; i < num_rows; i++)
    {
        for (j = 0; j < num_cols; j++)
	{
            element = matrix[i * num_cols + j];
            printf("%0.2f ", element);
        }
        printf("\n");
    }

    return;
}













