# Simple makefile for building PSO on xunil
# Author: Naga Kandasamy, May 5, 2020

CC		:= /usr/bin/gcc
CCFLAGS := -fopenmp -std=c99 -Wall -O3 
LDLIBS := -lm

all: pso

pso: pso.o pso_utils.o optimize_gold.o optimize_using_omp.o 
	$(CC) -o pso pso.o pso_utils.o optimize_gold.o optimize_using_omp.o $(LDLIBS)

pso.o: pso.c pso.h
	$(CC) -c pso.c $(CCFLAGS)

pso_utils.o: pso_utils.c pso.h
	$(CC) -c pso_utils.c $(CCFLAGS)

optimize_gold.o: optimize_gold.c pso.h
	$(CC) -c optimize_gold.c $(CCFLAGS)

optimize_using_omp.o: optimize_using_omp.c pso.h 
	$(CC) -c optimize_using_omp.c $(CCFLAGS)

clean: 
	rm pso *.o


