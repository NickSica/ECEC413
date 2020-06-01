#include "counting_sort.h"

__global__ void generate_bins_kernel(int *bin, int *input_array)
{
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bin[input_array[input_idx]]++;
    return;
}

__global__ void counting_sort_kernel(int *bin, int *input_array, int *sorted_array, int num_elements)
{
    __shared__ int bin_s[(MAX_VALUE + 1) * 2];
    int num_bins = blockDim.x;
    int bin_num = blockIdx.x * num_bins + threadIdx.x;

    bin_s[bin_num] = bin[bin_num];
    bin_s[num_bins + bin_num] = bin_s[bin_num] + bin_s[bin_num - 1];

    __syncthreads();
    
    // Calculate starting indices in output array for storing sorted elements. 
    // Use inclusive scan of the bin elements.
    int ping_pong = 0;
    for(int i = 2; i < blockDim.x; i = i * 2)
    {
	int new_idx = num_bins * ping_pong + threadIdx.x;
	int old_idx = num_bins * ~ping_pong + threadIdx.x;
	if(threadIdx.x >= i)
	    bin_s[new_idx] = bin_s[old_idx] + bin_s[old_idx - i];

	ping_pong = ~ping_pong;
	__syncthreads();
    }
    	    
    // Generate sorted array
    for(int j = bin_s[bin_num - 1]; j < bin_s[bin_num]; j++)
	sorted_array[j] = bin_num;

    return;
}
