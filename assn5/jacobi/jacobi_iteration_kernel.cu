#include "jacobi_iteration.h"
//#include <__clang_cuda_builtin_vars.h>

__global__ void jacobi_iteration_kernel_naive(float *A, float *B, unsigned int num_cols, unsigned int num_rows, float *x, double *ssd)
{
    __shared__ double s_ssd[MATRIX_SIZE];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
 
    float old_val = x[row];
    double sum = -A[row * num_cols + row] * x[row];
    for(int j = 0; j < num_cols; j++)
	sum += A[row * num_cols + j] * x[j];

    __syncthreads();
    
    x[row] = (B[row] - sum) / A[row * num_cols + row];

    // Check for convergence and update the unknowns.
    double val_diff = x[row] - old_val;
    double priv_ssd = val_diff * val_diff;
    s_ssd[row] = priv_ssd;

    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1)
    {
	if(row < stride)
	    s_ssd[row] += s_ssd[row + stride];
	__syncthreads();
    }
    
    if(row == 0)
	*ssd = s_ssd[0];
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *B, unsigned int num_cols, unsigned int num_rows, float *x, double *ssd)
{
    __shared__ double s_ssd[MATRIX_SIZE];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float diag_val = A[row * num_cols + row];
    double sum = -diag_val * x[row];
    for(int j = 0; j < num_cols; j++)
	sum += A[j * num_cols + row] * x[j];
    
    float new_val = (B[row] - sum) / diag_val;
    double val_diff = new_val - x[row];
    double priv_ssd = val_diff * val_diff;
    s_ssd[row] = priv_ssd;

    __syncthreads();
    
    // Check for convergence and update the unknowns.
    for(int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1)
    {
	if(row < stride)
	    s_ssd[row] += s_ssd[row + stride];
	__syncthreads();
    }
    
    if(row == 0)
	*ssd = s_ssd[0];

    x[row] = new_val;
    
    return;
}

__global__ void row_to_col_major_kernel(float *A, unsigned int num_cols, unsigned int num_rows, float *col_major_A)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    col_major_A[col * num_rows + row] = A[row * num_cols + col];
}
