/*
 * CUDA Matrix Multiplication: C = A x B.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cassert>
#include <cfloat>

#include "utilities.h"

const int numVerifications = 100;

__global__ void matrixMultiply(const float *A, const float *B, float *C, 
    int numElements) {

    int indexX = blockDim.x*blockIdx.x + threadIdx.x;
    int indexY = blockDim.y*blockIdx.y + threadIdx.y;

    if (indexX < numElements && indexY < numElements) {
        float sum = 0.0;
        for (int i = 0; i < numElements; i++)
            sum += A[indexX*numElements+i] * B[i*numElements+indexY];
        C[indexX*numElements+indexY] = sum;
        int index = indexX*numElements+indexY;
        // printf("cuda: [%d][%d] [%d] %.4f %.4f\n", indexX, indexY, index, 
        //     C[indexX*numElements+indexY], sum);
    }
}

void verify(const float *A, const float *B, const float *C, int numElements, 
    int p, int q) {

    // Verify that the result for C[p][q] is accurate
    float sum = 0.0;
    for (int i = 0; i < numElements; ++i)
        sum += A[p*numElements+i] * B[i*numElements+q];

    int index = p*numElements + q;
    printf("Verification PASSED for [%d][%d] %.8f / %.8f\n", p, q, C[index], sum);
    assert(fabsf(C[index] - sum) < CUDA_FLT_EPSILON);
}

int main(int argc, char** argv)
{
    // Print the vector length to be used, and compute its size
    int numElements = 256;
    printf("[Matrix Multiplication of %dx%d elements]\n", numElements, numElements);

    size_t size = numElements * numElements;

    // Allocate the host input vector A, B, C
    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size]();

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements*numElements; ++i) {
        h_A[i] = randomFloat();
        h_B[i] = randomFloat();
    }
    // // Verify results
    // for (int i = 0; i < numVerifications; i++) {
    //     int p = randomInt(numElements);
    //     int q = randomInt(numElements);
    //     verify(h_A, h_B, h_C, numElements, p, q);
    // }

    // CUDA 
    cudaError_t err = cudaSuccess;
    int d_size = size * sizeof(float);

    // Allocate device vectors A, B, C
    float *d_A, *d_B, *d_C;
    err = cudaMalloc((void**)&d_A, d_size);
    assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_B, d_size);
    assert(err == cudaSuccess);
    err = cudaMalloc((void**)&d_C, d_size);
    assert(err == cudaSuccess);

    // Initialize the input device vectors
    err = cudaMemcpy(d_A, h_A, d_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_B, h_B, d_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Launch matrix multiply on the GPU
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numElements+threadsPerBlock.x-1)/threadsPerBlock.x, 
        (numElements+threadsPerBlock.y-1)/threadsPerBlock.y);
    printf("Matrix Multiply on CPU launch\n");

    // Time the kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto t1 = Clock::now();

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaCheckErrors("kernel launch failure");

    auto t2 = Clock::now();
    auto dur = t2 - t1;
    unsigned long nanos = std::chrono::nanoseconds(dur).count();
    printf("Matrix Multiply on GPU time: %ld ns\n", nanos);

    // Copy the results back
    err = cudaMemcpy(h_C, d_C, d_size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    // Free devie vectors
    err = cudaFree(d_A);
    assert(err == cudaSuccess);
    err = cudaFree(d_B);
    assert(err == cudaSuccess);
    err = cudaFree(d_C);
    assert(err == cudaSuccess);

    // Verify results
    for (int i = 0; i < numVerifications; i++) {
        int p = randomInt(numElements);
        int q = randomInt(numElements);
        verify(h_A, h_B, h_C, numElements, p, q);
    }
    // for (int i = 0; i < numElements; i++) {
    //     for (int j = 0; j < numElements; j++) {
    //         verify(h_A, h_B, h_C, numElements, i, j);
    //     }
    // }

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
