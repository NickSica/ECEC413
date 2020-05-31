
//#include <__clang_cuda_builtin_vars.h>
__constant__ float kernel_c[2 * HALF_WIDTH + 1];

__global__ void convolve_rows_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows)
{
    int i1;
    int j1, j2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    j1 = x - HALF_WIDTH;
    j2 = x + HALF_WIDTH;
    
    // Clamp at the edges of the matrix
    if (j1 < 0) 
	j1 = 0;
	
    if (j2 >= num_cols) 
	j2 = num_cols - 1;
    
    // Obtain relative position of starting element from element being convolved
    i1 = j1 - x;
    
    // Obtain operating width of the kernel
    j1 = j1 - x + HALF_WIDTH; 
    j2 = j2 - x + HALF_WIDTH;
    
    // Convolve along row
    result[y * num_cols + x] = 0.0f;
    for(int i = i1, j = j1; j <= j2; j++, i++)
	result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + i];
    
    return;
}

__global__ void convolve_columns_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows)
{
    int i1;
    int j1, j2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    j1 = y - HALF_WIDTH;
    j2 = y + HALF_WIDTH;
    
    // Clamp at the edges of the matrix
    if (j1 < 0) 
	j1 = 0;
    
    if (j2 >= num_rows) 
	j2 = num_rows - 1;
    
    // Obtain relative position of starting element from element being convolved
    i1 = j1 - y; 
    
    // Obtain the operating width of the kernel
    j1 = j1 - y + HALF_WIDTH;
    j2 = j2 - y + HALF_WIDTH;
	
    // Convolve along column
    result[y * num_cols + x] = 0.0f;
    for(int i = i1, j = j1; j <= j2; j++, i++)
	result[y * num_cols + x] += kernel[j] * input[y * num_cols + x + (i * num_cols)];
    
    return;
}

__global__ void convolve_rows_kernel_optimized(float *result, float *input, int num_cols, int num_rows)
{
    /*
    const int num_cols_total = THREAD_BLOCK_SIZE + HALF_WIDTH * 2;
    __shared__ float input_s[num_cols_total * THREAD_BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load the left halo elements of previous tile
    int left_halo_idx = (blockIdx.x - 1) * blockDim.x + threadIdx.x;    
    if(threadIdx.x >= (blockDim.x - HALF_WIDTH))
    {
	if(left_halo_idx < 0)
	    input_s[threadIdx.x - (blockDim.x - HALF_WIDTH)] = 0.0;
	else
	    input_s[threadIdx.x - (blockDim.x - HALF_WIDTH)] = input[left_halo_idx];
    }

    // Load the center elements for the tile
    if(i < num_cols)
	input_s[HALF_WIDTH + threadIdx.x] = input[i];
    else
	input_s[HALF_WIDTH + threadIdx.x] = 0.0;
	
    // Load the right halo elements of previous tile
    int right_halo_idx = (blockIdx.x + 1) * blockDim.x + threadIdx.x;    
    if(threadIdx.x < HALF_WIDTH)
    {
	if(right_halo_idx >= num_cols)
	    input_s[threadIdx.x + (blockDim.x + HALF_WIDTH)] = 0.0;
	else
	    input_s[threadIdx.x + (blockDim.x + HALF_WIDTH)] = input[right_halo_idx];
    }

    __syncthreads();
    */
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int col_start = x - HALF_WIDTH;
    int col_end = x + HALF_WIDTH;
    
    // Clamp at the edges of the matrix
    if(col_start < 0) 
	col_start = 0;
	
    if(col_end >= num_cols) 
	col_end = num_cols - 1;
    
    // Obtain relative position of starting element from element being convolved
    int row = col_start - x;
    
    // Obtain operating width of the kernel
    col_start = col_start - x + HALF_WIDTH; 
    col_end = col_end - x + HALF_WIDTH;
    
    // Convolve along row
    float res = 0.0f;
    for(int j = col_start; j <= col_end; j++)
    {
	res += kernel_c[j] * input[y * num_cols + x + row];
	row++;
    }
    result[y * num_cols + x] = res;
    return;
}

__global__ void convolve_columns_kernel_optimized(float *result, float *input, int num_cols, int num_rows)
{
    /*
    __shared__ float input_s[THREAD_BLOCK_SIZE + HALF_WIDTH * 2];

    // Load the upper halo elements of previous tile
    int upper_halo_idx = (blockIdx.y - 1) * blockDim.y + threadIdx.y;    
    if(threadIdx.y >= (blockDim.y - HALF_WIDTH))
    {
	if(upper_halo_idx < 0)
	    input_s[threadIdx.y - (blockDim.y - HALF_WIDTH)] = 0.0;
	else
	    input_s[threadIdx.y - (blockDim.y - HALF_WIDTH)] = input[upper_halo_idx];
    }

    // Load the center elements for the tile
    if(x < num_rows)
	input_s[HALF_WIDTH + threadIdx.y] = input[x];
    else
	input_s[HALF_WIDTH + threadIdx.y] = 0.0;
	
    // Load the lower halo elements of previous tile
    int lower_halo_idx = (blockIdx.y + 1) * blockDim.y + threadIdx.y;    
    if(threadIdx.y < HALF_WIDTH)
    {
	if(lower_halo_idx >= num_rows)
	    input_s[threadIdx.y + (blockDim.y + HALF_WIDTH)] = 0.0;
	else
	    input_s[threadIdx.y + (blockDim.y + HALF_WIDTH)] = input[lower_halo_idx];
    }

    __syncthreads();
    */
    int row_start, row_end;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    row_start = y - HALF_WIDTH;
    row_end = y + HALF_WIDTH;
    
    // Clamp at the edges of the matrix
    if(row_start < 0) 
	row_start = 0;
    
    if(row_end >= num_rows) 
	row_end = num_rows - 1;
    
    // Obtain relative position of starting element from element being convolved
    int col = row_start - y; 
    
    // Obtain the operating width of the kernel
    row_start = row_start - y + HALF_WIDTH;
    row_end = row_end - y + HALF_WIDTH;
	
    // Convolve along column
    float res = 0.0f;
    for(int j = row_start; j <= row_end; j++)
    {
	res += kernel_c[j] * input[y * num_cols + x + (col * num_cols)];
	col++;
    }

    result[y * num_cols + x] = res;
    return;
}








