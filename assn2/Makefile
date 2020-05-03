build:
	gcc -O3 -Wall -lm -lpthread jacobi_solver.c compute_gold.c -o jacobi_solver
	gcc -O3 -Wall -lm -lpthread gauss_eliminate.c compute_gold.c -o gauss_eliminate

jacobi: build
	./jacobi_solver 512

gauss: build
	./gauss_eliminate 512

clean:
	-rm jacobi_solver
	-rm gauss_eliminate
