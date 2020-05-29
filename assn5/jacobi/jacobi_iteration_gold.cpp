/* Reference code for solving the equation by jacobi by iteration method */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi_iteration.h"

void compute_gold(const matrix_t A, matrix_t x, const matrix_t B)
{
    unsigned int i, j, k;
    unsigned int num_rows = A.num_rows;
    unsigned int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values */
    matrix_t new_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);      
    
    /* Initialize current jacobi solution */
    for (i = 0; i < num_rows; i++)
        x.elements[i] = B.elements[i];

    /* Perform Jacobi iteration */
    unsigned int done = 0;
    double ssd, mse;
    unsigned int num_iter = 0;
    
    while (!done) {
        for (i = 0; i < num_rows; i++) {
            double sum = 0.0;
            for (j = 0; j < num_cols; j++) {
                if (i != j)
                    sum += A.elements[i * num_cols + j] * x.elements[j];
            }
           
            /* Update values for the unkowns for the current row. */
            new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
        }

        /* Note: you can optimize the above nested loops by removing the branch 
         * statement within the j loop. The rewritten code is as follows: 
         *
         * for (i = 0; i < num_rows; i++){
         *      double sum = -A.elements[i * num_cols + i] * ref_x.elements[i];
         *      for (j = 0; j < num_cols; j++)
         *          sum += A.elements[i * num_cols + j] * ref_x.elements[j];
         * }
         *
         * new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
         *
         * I recommend using this code snippet within your GPU kernel implementation 
         * to eliminate branch divergence between the threads.
         * 
         * */

        /* Check for convergence and update the unknowns. */
        ssd = 0.0; 
        for (i = 0; i < num_rows; i++) {
            ssd += (new_x.elements[i] - x.elements[i]) * (new_x.elements[i] - x.elements[i]);
            x.elements[i] = new_x.elements[i];
        }
        num_iter++;
        mse = sqrt (ssd); /* Mean squared error. */
        printf("Iteration: %d. MSE = %f\n", num_iter, mse); 
        
        if (mse <= THRESHOLD)
            done = 1;
    }

    printf("\nConvergence achieved after %d iterations \n", num_iter);
    free(new_x.elements);
}
    
/* Display statistics related to the Jacobi solution */
void display_jacobi_solution(const matrix_t A, const matrix_t x, const matrix_t B)
{
	double diff = 0.0;
	unsigned int num_rows = A.num_rows;
    unsigned int num_cols = A.num_columns;
	
    for (unsigned int i = 0; i < num_rows; i++) {
		double line_sum = 0.0;
		for (unsigned int j = 0; j < num_cols; j++){
			line_sum += A.elements[i * num_cols + j] * x.elements[j];
		}
		
        diff += fabsf(line_sum - B.elements[i]);
	}

	printf("Average diff between LHS and RHS: %f \n", diff/(float)num_rows);
    return;
}

