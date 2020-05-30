#include "jacobi_iteration.h"
//#include <__clang_cuda_builtin_vars.h>

__global__ void jacobi_iteration_kernel_naive(float *A, float *B, unsigned int num_cols, unsigned int num_rows, float *x, double *ssd)
{
    int row = threadIdx.x; // blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double s_ssd[MATRIX_SIZE];

    float old_val = x[row];
    double sum = -A[row * num_cols + row] * x[row];
    for(int j = 0; j < num_cols; j++)
	sum += A[row * num_cols + j] * x[j];
    x[row] = (B[row] - sum) / A[row * num_cols + row];

    // Check for convergence and update the unknowns.
    double priv_ssd = (x[row] - old_val) * (x[row] - old_val);
    printf("Priv SSD: %f\n", priv_ssd);
    s_ssd[row] = priv_ssd;
    __syncthreads();
    for(int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1)
    {
	if(threadIdx.x < stride)
	    s_ssd[threadIdx.x] += s_ssd[threadIdx.x + stride];
	__syncthreads();
    }

    if(threadIdx.x == 0)
	*ssd = s_ssd[0];

    printf("KERNEL SSD: %f", *ssd);
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *B, unsigned int num_cols, unsigned int num_rows, float *x, double *ssd)
{
    /*
    int curr_row, curr_col;
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Obtain row number of pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Obtain column number of pixel

    for (int i = 0; i < num_rows; i++)
    {
	double sum = -A[i * num_cols + i] * x[i];
	for (int j = 0; j < num_cols; j++)
	    sum += A[i * num_cols + j] * x[j];
	
	new_x[i] = (B[i] - sum) / A[i * num_cols + i];
    }

    // Check for convergence and update the unknowns.
    for (int i = 0; i < A.num_rows; i++)
    {
	ssd += (new_x.elements[i] - x.elements[i]) * (new_x.elements[i] - x.elements[i]);
	x.elements[i] = new_x.elements[i];
    }
    */
    return;
}





