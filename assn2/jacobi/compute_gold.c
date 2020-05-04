/* Reference code for solving the equation by jacobi iteration method */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi_solver.h"

void compute_gold(const matrix_t A, matrix_t x, const matrix_t B, int max_iter)
{
    int i, j;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      
    
    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++)
        x.elements[i] = B.elements[i];

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse;
    int num_iter = 0;
    
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
         */

        /* Check for convergence and update the unknowns. */
        ssd = 0.0; 
        for (i = 0; i < num_rows; i++) {
            ssd += (new_x.elements[i] - x.elements[i]) * (new_x.elements[i] - x.elements[i]);
            x.elements[i] = new_x.elements[i];
        }
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        //fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
        
        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
    }

    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");

    free(new_x.elements);
}
    
/* Display statistic related to the Jacobi solution */
void display_jacobi_solution (const matrix_t A, const matrix_t x, const matrix_t B)
{
	double diff = 0.0;
	int num_rows = A.num_rows;
    int num_cols = A.num_columns;
    int i, j;
    double line_sum;
	
    for (i = 0; i < num_rows; i++) {
		line_sum = 0.0;
		for (j = 0; j < num_cols; j++){
			line_sum += A.elements[i * num_cols + j] * x.elements[j];
		}
		
        diff += fabsf(line_sum - B.elements[i]);
	}

	fprintf(stderr, "Average diff between LHS and RHS %f\n", diff/(float)num_rows);
    return;
}

