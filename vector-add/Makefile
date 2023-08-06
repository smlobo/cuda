# Makefile for vectorAdd
# * GPU
# * GPU with a Grid Stride loop
# * CPU

CPP_FLAGS = -O -std=c++14

INCLUDE_DIR = -I../common

COMMON_FILES = ../common/utilities.cc

all: vectorAddGPU vectorAddGPUGridStride vectorAddCPU

.PHONY: vectorAddGPU
vectorAddGPU:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddGPU.cu -o \
	vectorAddGPU

.PHONY: vectorAddGPUGridStride
vectorAddGPUGridStride:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddGPUGridStride.cu \
	-o vectorAddGPUGridStride

.PHONY: vectorAddCPU
vectorAddCPU:
	g++ $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddCPU.cc -o \
	vectorAddCPU

