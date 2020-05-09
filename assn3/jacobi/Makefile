build:
	gcc --std=c99 -O3 -Wall -lm -fopenmp jacobi_solver.c compute_gold.c -o jacobi_solver

run: build
	OMP_NUM_THREADS=4 ./jacobi_solver 512

clean:
	-rm jacobi_solver
