/* Write GPU code to perform the step(s) involved in counting sort.
 Add additional kernels and device functions as needed. */
__global__ void counting_sort_kernel(int *bin, int *input_array, int *sorted_array, int num_elements)
{
    // Step 1: Compute histogram and generate bin for each element within the range
    int bin_num = blockIdx.x * blockDim.x + threadIdx.x;
    bin[input_array[bin_num]]++;

    __syncthreads();

    int old_bin = bin[bin_num - 1];

    __syncthreads();

    // Step 2: Calculate starting indices in output array for storing sorted elements. 
    // Use inclusive scan of the bin elements.
    if(bin_num != 0)
        bin[bin_num] = old_bin + bin[bin_num];

    __syncthreads();

    // Step 3: Generate sorted array
    for(int j = bin[bin_num - 1]; j < bin[bin_num]; j++)
	sorted_array[j] = bin_num;

    return;
}
