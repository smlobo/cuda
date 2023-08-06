/*
 * CUDA Matrix Multiplication: C = A x B.
 * Using shared memory common to a Thread Block
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cassert>
#include <cfloat>

#include "utilities.h"

const int numVerifications = 100;
const int numElements = 1024;
const int numThreads = 16;

void verify(const float*, const float*, const float*, int, int, int, int);

__global__ void matrixMultiply(const float *A, const float *B, float *C, 
    int numElements) {

    // Cache of shared memory for a Thread Block
    __shared__ float cacheA[numThreads][numThreads];
    __shared__ float cacheB[numThreads][numThreads];

    int indexX = blockDim.x*blockIdx.x + threadIdx.x;
    int indexY = blockDim.y*blockIdx.y + threadIdx.y;
    int index = indexX*numElements+indexY;

    if (indexX < numElements && indexY < numElements) {
        float sum = 0.0;

        // Iterate over unique blocks (move stencil to the next block)
        int numBlocks = numElements/numThreads;
        for (int i = 0; i < numBlocks; i++) {

            // Load data into shared memory
            int xA = indexX;
            int yA = i * numThreads + threadIdx.y;
            cacheA[threadIdx.x][threadIdx.y] = A[xA*numElements + yA];
            int xB = i * numThreads + threadIdx.x;
            int yB = indexY;
            cacheB[threadIdx.x][threadIdx.y] = B[xB*numElements + yB];

            // Synchronize with remaining threads in block; cache filled
            __syncthreads();

            // Accumulate sum for the current cache 
            for (int j = 0; j < numThreads; j++)
                sum += cacheA[threadIdx.x][j] * cacheB[j][threadIdx.y];

            // Synchronize before cache gets overwritten
            __syncthreads();
        }

        C[index] = sum;
    }
}

int main(int argc, char** argv)
{
    // Print the vector length to be used, and compute its size
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

    // CUDA timing
    typedef std::chrono::high_resolution_clock Clock;
    auto tStart = Clock::now();

    // Initialize the input device vectors
    err = cudaMemcpy(d_A, h_A, d_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    err = cudaMemcpy(d_B, h_B, d_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Launch matrix multiply on the GPU
    dim3 threadsPerBlock(numThreads, numThreads);
    int numBlocks = (numElements+numThreads-1)/numThreads;
    dim3 blocksPerGrid(numBlocks, numBlocks);
    printf("Matrix Multiply Shared on GPU launch\n");

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaCheckErrors("kernel launch failure");

    // Copy the results back
    err = cudaMemcpy(h_C, d_C, d_size, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    auto duration = Clock::now() - tStart;
    printf("Matrix Multiply Shared on GPU time: %s\n", 
        nanoToString(std::chrono::nanoseconds(duration).count()));

    // Free device vectors
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
        verify(h_A, h_B, h_C, numElements, p, q, i);
    }
    printf("Verified for %d random elements\n", numVerifications);
    // Verify all results
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
