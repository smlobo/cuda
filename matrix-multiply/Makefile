# Makefile for matrixMultiply
# * GPU
# * CPU

# CPP_FLAGS = -O -std=c++14
CPP_FLAGS = -g -std=c++14

INCLUDE_DIR = -I../common

COMMON_FILES = ../common/utilities.cc

all: matrixMultiplyGPU matrixMultiplyCPU

.PHONY: matrixMultiplyGPU
matrixMultiplyGPU:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) matrixMultiplyGPU.cu \
	-o matrixMultiplyGPU

.PHONY: matrixMultiplyCPU
matrixMultiplyCPU:
	g++ $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) matrixMultiplyCPU.cc \
	-o matrixMultiplyCPU
