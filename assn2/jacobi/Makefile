build:
	gcc -std=gnu99 -O3 -Wall -lm -lpthread jacobi_solver.c compute_gold.c -o jacobi_solver

run: build
	./jacobi_solver 512

clean:
	-rm jacobi_solver
