build:
	gcc --std=gnu99 -O3 -Wall -lm -fopenmp gauss_eliminate.c compute_gold.c -o gauss_eliminate

run: build
	OP_NUM_THREADS=4 ./gauss_eliminate 2048 

clean:
	-rm gauss_eliminate
