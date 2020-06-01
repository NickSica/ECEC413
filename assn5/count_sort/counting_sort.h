#ifndef _COUNT_SORT_H_
#define _COUNT_SORT_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* Do not change the range value */
#define MIN_VALUE 0 
#define MAX_VALUE 255

/* Uncomment to spit out debug info */
// #define DEBUG

extern "C" int counting_sort_gold(int *, int *, int, int);
int rand_int(int, int);
void print_histogram(int *, int, int);
void print_array(int *, int);
void print_min_and_max_in_array(int *, int);
void compute_on_device(int *, int *, int, int);
void check_for_error(const char *);
int check_if_sorted(int *, int);
int compare_results(int *, int *, int);

#endif
