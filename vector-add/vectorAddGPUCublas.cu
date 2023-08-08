/**
 * Vector addition: C = A + B.
 */

#include <cstdio>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cublas_v2.h>

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
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
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

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cublasCreate(&cublasH);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(cublasH, stream);

    /* step 2: copy data to device */
    float *d_A, *d_B;
    cudaMalloc(reinterpret_cast<void **>(&d_A), size);
    cudaMalloc(reinterpret_cast<void **>(&d_B), size);

    // Time the GPU code
    typedef std::chrono::high_resolution_clock Clock;
    auto tStart = Clock::now();

    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    /* step 3: compute */
    const float alpha = 1.0;
    const int incx = 1;
    const int incy = 1;
    cublasSaxpy(cublasH, numElements, &alpha, d_A, incx, d_B, incy);

    /* step 4: copy data to host */
    cudaMemcpyAsync(h_C, d_B, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    auto duration = Clock::now() - tStart;
    printf("Vector Add on GPU with libcublas time: %s\n", 
        nanoToString(std::chrono::nanoseconds(duration).count()));

    // Free device vectors
    err = cudaFree(d_A);
    assert(err == cudaSuccess);
    err = cudaFree(d_B);
    assert(err == cudaSuccess);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        // printf("%.4f %.4f %.4f\n", h_A[i], h_B[i], h_C[i]);
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

