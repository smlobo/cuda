# Makefile for vectorAdd
# * cuda
# * vanilla

CPP_FLAGS = -O -std=c++14

all: vectorAddCuda vectorAddVanilla

.PHONY: vectorAddCuda
vectorAddCuda:
	nvcc $(CPP_FLAGS) vectorAddCuda.cu -o vectorAddCuda

.PHONY: vectorAddVanilla
vectorAddVanilla:
	g++ $(CPP_FLAGS) vectorAddVanilla.cc -o vectorAddVanilla

