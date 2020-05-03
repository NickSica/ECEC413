build:
	gcc -O3 -Wall -lm -lpthread gauss_eliminate.c compute_gold.c -o gauss_eliminate

run: build
	./gauss_eliminate 512

clean:
	-rm saxpy

