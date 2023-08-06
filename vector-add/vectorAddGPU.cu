/**
 * Vector addition: C = A + B.
 */

#include <cstdio>
#include <iostream>
#include <chrono>
#include <cassert>

#include "utilities.h"

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 10000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

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

    // Copy the host input vectors A and B in host memory to the device input 
    // vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA vectorAdd kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

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

