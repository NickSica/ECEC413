build:
	gcc --std=gnu99 -O3 -Wall -lm -fopenmp gauss_eliminate.c compute_gold.c -o gauss_eliminate

run: build
	./gauss_eliminate 512

clean:
	-rm gauss_eliminate
