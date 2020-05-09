build:
	gcc --std=gnu99 -O3 -Wall -lm -lpthread jacobi/jacobi_solver.c jacobi/compute_gold.c -o jacobi_solver
	gcc --std=gnu99 -O3 -Wall -lm -lpthread gaussian/gauss_eliminate.c gaussian/compute_gold.c -o gauss_eliminate

jacobi: build
	./jacobi_solver 512

gauss: build
	./gauss_eliminate 512

clean:
	-rm jacobi_solver
	-rm gauss_eliminate
