# Makefile for vectorAdd
# * GPU
# * GPU with a Grid Stride loop
# * GPU chunked
# * GPU with libcublas
# * CPU

CPP_FLAGS = -O2 --std=c++20
#CPP_FLAGS = -O2 --std=c++20 -v -keeptmp

INCLUDE_DIR = -I../common

COMMON_FILES = ../common/utilities.cc

all: vectorAddGPU vectorAddGPUGridStride vectorAddGPUChunk vectorAddGPUCublas vectorAddCPU

.PHONY: vectorAddGPU
vectorAddGPU:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddGPU.cu -o \
	vectorAddGPU

.PHONY: vectorAddGPUGridStride
vectorAddGPUGridStride:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddGPUGridStride.cu \
	-o vectorAddGPUGridStride

.PHONY: vectorAddGPUChunk
vectorAddGPUChunk:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddGPUChunk.cu -o \
	vectorAddGPUChunk

.PHONY: vectorAddGPUCublas
vectorAddGPUCublas:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddGPUCublas.cu -o \
	vectorAddGPUCublas -L/usr/local/cuda/lib64 -lcublas

.PHONY: vectorAddCPU
vectorAddCPU:
	g++ $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) vectorAddCPU.cc -o \
	vectorAddCPU

