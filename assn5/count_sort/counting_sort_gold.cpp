#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// #define DEBUG

extern "C" int counting_sort_gold(int *, int *, int, int);
void print_histogram(int *, int, int);

/* Reference implementation of counting sort */
int counting_sort_gold(int *input_array, int *sorted_array, int num_elements, int range)
{
    /* Step 1: Compute histogram and generate bin for each element within the range */ 
    int i;
    int num_bins = range + 1;
    int *bin = (int *)malloc(num_bins * sizeof(int));    
    if (bin == NULL) {
        perror("malloc");
        return -1;
    }

    memset(bin, 0, num_bins); 
    for (i = 0; i < num_elements; i++)
        bin[input_array[i]]++;

#ifdef DEBUG
    print_histogram(bin, num_bins, num_elements);
#endif

    /* Step 2: Calculate starting indices in output array for storing sorted elements. 
     * Use inclusive scan of the bin elements. */
    for (i = 1; i < num_bins; i++)
        bin[i] = bin[i - 1] + bin[i];

#ifdef DEBUG
    print_histogram(bin, num_bins, num_elements);
#endif

    /* Step 3: Generate sorted array */
    int j;
    int start_idx = 0;
    for (i = 0; i < num_bins; i++) {
        for (j = start_idx; j < bin[i]; j++) {
            sorted_array[j] = i;
        }
        start_idx = bin[i];
    }

    return 0;
}

/* Print histogram contents */
void print_histogram(int *bin, int num_bins, int num_elements)
{
    int num_histogram_entries = 0;
    int i;

    for (i = 0; i < num_bins; i++) {
        printf("Bin %d: %d\n", i, bin[i]);
        num_histogram_entries += bin[i];
    }

    printf("Number of elements in the input array = %d \n", num_elements);
    printf("Number of histogram elements = %d \n", num_histogram_entries);

    return;
}

