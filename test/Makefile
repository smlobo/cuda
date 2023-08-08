# Makefile for "test"

CPP_FLAGS = -O2 -std=c++14
# CPP_FLAGS = -g -std=c++14

INCLUDE_DIR = -I../common

COMMON_FILES = ../common/utilities.cc

all: cudaKernel

.PHONY: cudaKernel
cudaKernel:
	nvcc $(CPP_FLAGS) $(INCLUDE_DIR) $(COMMON_FILES) cudaKernel.cu \
	-o cudaKernel

