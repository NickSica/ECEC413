#include <stdio.h>
#include <stdlib.h>

extern int compute_gold(float *, int);

/* Perform Gaussian elimination in place on the U matrix */
int compute_gold(float *U, int num_elements)
{
    int i, j, k;
    
    for (k = 0; k < num_elements; k++) {
        for (j = (k + 1); j < num_elements; j++) {   /* Reduce the current row. */
            if (U[num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                return -1;
            }            
            U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);	/* Division step */
        }

        U[num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */ 
        
        for (i = (k + 1); i < num_elements; i++) {
            for (j = (k + 1); j < num_elements; j++)
                U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step */
            
            U[num_elements * i + k] = 0;
        }
    }
    
    return 0;
}
