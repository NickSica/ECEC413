/* Host-side code to perform counting sort 
 * Author: Naga Kandasamy
 * Date modified: May 27, 2020
 * 
 * Compile as follows: make clean && make
 */

#include "counting_sort.h"
#include "counting_sort_kernel.cu"

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s num-elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_elements = atoi(argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate input array with random integers between [0, RANGE] */
    printf("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *)malloc(num_elements * sizeof(int));
    if (input_array == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array(input_array, num_elements);
    print_min_and_max_in_array(input_array, num_elements);
#endif

    /* Sort elements in input array using reference implementation. 
     * The result is placed in sorted_array_reference. */
    printf("\nSorting array on CPU\n");
    int status;
    sorted_array_reference = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_reference == NULL) {
        perror("malloc"); 
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_reference, 0, num_elements);
    status = counting_sort_gold(input_array, sorted_array_reference, num_elements, range);
    if (status == -1) {
        exit(EXIT_FAILURE);
    }

    status = check_if_sorted(sorted_array_reference, num_elements);
    if (status == -1) {
        printf("Error sorting the input array using the reference code\n");
        exit(EXIT_FAILURE);
    }

    printf("Counting sort was successful on the CPU\n");

#ifdef DEBUG
    print_array(sorted_array_reference, num_elements);
#endif

    /* FIXME: Write function to sort elements in the array in parallel fashion. 
     * The result should be placed in sorted_array_mt. */
    printf("\nSorting array on GPU\n");
    sorted_array_d = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_d == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_d, 0, num_elements);
    compute_on_device(input_array, sorted_array_d, num_elements, range);

    /* Check the two results for correctness */
    printf("\nComparing CPU and GPU results\n");
    status = compare_results(sorted_array_reference, sorted_array_d, num_elements);
    if (status == 0)
        printf("Test passed\n");
    else
        printf("Test failed\n");

    exit(EXIT_SUCCESS);
}


/* GPU implementation of counting sort */
void compute_on_device(int *input_array, int *sorted_array, int num_elements, int range)
{
    
    int num_bins = range + 1;
    int *bin = (int *)malloc(num_bins * sizeof(int));
    memset(bin, 0, num_bins);

    int *d_bin;
    int *d_input_array;
    int *d_sorted_array;
    int input_size = num_elements * sizeof(int);
    int bin_size = num_bins * sizeof(int);
    cudaMalloc(&d_bin, bin_size);
    cudaMalloc(&d_input_array, input_size);
    cudaMalloc(&d_sorted_array, input_size);
    check_for_error("CUDA Malloc Failure");
    
    cudaMemcpy(d_bin, bin, bin_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sorted_array, sorted_array, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_array, input_array, input_size, cudaMemcpyHostToDevice);
    check_for_error("CUDA Memcpy To Device Failure");

    dim3 bins_threads(num_elements, 1, 1);
    dim3 grid(1, 1, 1);
    
    generate_bins_kernel<<<grid, bins_threads>>>(d_bin, d_input_array);
    cudaDeviceSynchronize();
    check_for_error("Bin Generation Kernel Launch Failure");

    dim3 threads(num_bins, 1, 1);

    counting_sort_kernel<<<grid, threads, 2 * bin_size>>>(d_bin, d_input_array, d_sorted_array, num_elements);
    cudaDeviceSynchronize();
    check_for_error("Counting Sort Kernel Launch Failure");

    cudaMemcpy(sorted_array, d_sorted_array, input_size, cudaMemcpyDeviceToHost);
    check_for_error("CUDA Memcpy From Device Failure");

    print_array(input_array, num_elements);
    print_array(sorted_array, num_elements);
    
    free(bin);
    cudaFree(d_bin);
    cudaFree(d_sorted_array);
    return;
}

/* Check for errors reported by the CUDA run time */
void check_for_error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
	printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    
    return;
} 

/* Check if array is sorted */
int check_if_sorted(int *array, int num_elements)
{
    int status = 0;
    int i;
    for (i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical */ 
int compare_results(int *array_1, int *array_2, int num_elements)
{
    int status = 0;
    int i;
    for (i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Return random integer between [min, max] */ 
int rand_int(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (int)floorf(min + (max - min) * r);
}

/* Print given array */
void print_array(int *this_array, int num_elements)
{
    printf("Array: ");
    int i;
    for (i = 0; i < num_elements; i++)
        printf("%d ", this_array[i]);
    
    printf("\n");
    return;
}

/* Return min and max values in given array */
void print_min_and_max_in_array(int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf("Minimum value in the array = %d\n", current_min);
    printf("Maximum value in the array = %d\n", current_max);
    return;
}


