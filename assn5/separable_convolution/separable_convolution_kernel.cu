//#include <__clang_cuda_builtin_vars.h>
__global__ void convolve_rows_kernel_naive(float *result, float *input, float *kernel, \
					   int num_cols, int num_rows, int half_width)
{
    int i1;
    int j1, j2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    j1 = x - half_width;
    j2 = x + half_width;
    
    // Clamp at the edges of the matrix
    if (j1 < 0) 
	j1 = 0;
	
    if (j2 >= num_cols) 
	j2 = num_cols - 1;
    
    // Obtain relative position of starting element from element being convolved
    i1 = j1 - x;
    
    // Obtain operating width of the kernel
    j1 = j1 - x + half_width; 
    j2 = j2 - x + half_width;
    
    // Convolve along row
    result[y * num_cols + x] = 0.0f;
    for(int i = i1, j = j1; j <= j2; j++, i++)
	result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + i];
    
    return;
}

__global__ void convolve_columns_kernel_naive(float *result, float *input, float *kernel, \
					      int num_cols, int num_rows, int half_width)
{
    int i1;
    int j1, j2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    j1 = y - half_width;
    j2 = y + half_width;
    
    // Clamp at the edges of the matrix
    if (j1 < 0) 
	j1 = 0;
    
    if (j2 >= num_rows) 
	j2 = num_rows - 1;
    
    // Obtain relative position of starting element from element being convolved
    i1 = j1 - y; 
    
    // Obtain the operating width of the kernel
    j1 = j1 - y + half_width;
    j2 = j2 - y + half_width;
	
    // Convolve along column
    result[y * num_cols + x] = 0.0f;
    for(int i = i1, j = j1; j <= j2; j++, i++)
	result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + (i * num_cols)];
    
    return;
}

__global__ void convolve_rows_kernel_optimized()
{
    return;
}

__global__ void convolve_columns_kernel_optimized()
{
    return;
}
