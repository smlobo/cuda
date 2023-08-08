/**
 * Vector addition: C = A + B.
 */

#include <cstdio>
#include <iostream>
#include <chrono>
#include <cassert>

#include "utilities.h"

const int numElements = 1000000;
const int chunkSize = numElements/100;
const size_t size = chunkSize * sizeof(float);

__global__ void
vectorAdd(const float *A, const float *B, float *C, int n) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void doSubVectorAdd(float* h_A, float* h_B, float* h_C, float* d_A, 
    float* d_B, float* d_C) {

    cudaError_t err = cudaSuccess;

    for (int i = 0; i < numElements; i+=chunkSize) {
        // Copy the host input vectors A and B in host memory to the device input 
        // vectors in device memory
        // printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A+i, size, cudaMemcpyHostToDevice);
        assert(err == cudaSuccess);
        err = cudaMemcpy(d_B, h_B+i, size, cudaMemcpyHostToDevice);
        assert(err == cudaSuccess);

        // Launch the Vector Add CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid =(chunkSize + threadsPerBlock - 1) / threadsPerBlock;
        // printf("CUDA vectorAdd kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, chunkSize);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        // printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(h_C+i, d_C, size, cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
    }
}

int main(void) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used
    printf("[Vector addition of %dM elements]\n", numElements/1000000);

    // Allocate the host input vector A, B, C
    float *h_A = new float[numElements];
    float *h_B = new float[numElements];
    float *h_C = new float[numElements];

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = randomFloat();
        h_B[i] = randomFloat();
    }

    // Allocate the device input vector A, B, C
    float *d_A, *d_B, *d_C;
    err = cudaMalloc((void **)&d_A, size);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_B, size);
    assert(err == cudaSuccess);
    err = cudaMalloc((void **)&d_C, size);
    assert(err == cudaSuccess);

    // Time the GPU code
    typedef std::chrono::high_resolution_clock Clock;
    auto tStart = Clock::now();

    doSubVectorAdd(h_A, h_B, h_C, d_A, d_B, d_C);

    auto duration = Clock::now() - tStart;
    printf("Vector Add on GPU time: %s\n", 
        nanoToString(std::chrono::nanoseconds(duration).count()));

    // Free device vectors
    err = cudaFree(d_A);
    assert(err == cudaSuccess);
    err = cudaFree(d_B);
    assert(err == cudaSuccess);
    err = cudaFree(d_C);
    assert(err == cudaSuccess);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > CUDA_FLT_EPSILON)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

