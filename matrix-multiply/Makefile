# Makefile for matrixMultiply
# * GPU
# * GPU with SM shared memory
# * CPU

NVCC = nvcc
# NVCC = /usr/local/cuda-13.1/bin/nvcc

CPP_FLAGS = -O2 -std=c++14
# CPP_FLAGS = -g -std=c++14

INCLUDE_DIR = -I../common

COMMON_FILES = verify.cc ../common/utilities.cc

all: matrixMultiplyGPU matrixMultiplyGPUShared matrixMultiplyCPU

.PHONY: matrixMultiplyGPU
matrixMultiplyGPU:
	$(NVCC) $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) matrixMultiplyGPU.cu \
	-o matrixMultiplyGPU

.PHONY: matrixMultiplyGPUShared
matrixMultiplyGPUShared:
	$(NVCC) $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) \
	matrixMultiplyGPUShared.cu -o matrixMultiplyGPUShared

.PHONY: matrixMultiplyCPU
matrixMultiplyCPU:
	g++ $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) matrixMultiplyCPU.cc \
	-o matrixMultiplyCPU

clean:
	rm -rf matrixMultiplyGPU matrixMultiplyGPUShared matrixMultiplyCPU

