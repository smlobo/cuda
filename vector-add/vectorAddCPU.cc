/**
 * Vector addition: C = A + B.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#include "utilities.h"

void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    for (int i = 0; i < numElements; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
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

    // Launch the Vanilla Vector Add CUDA Kernel
    printf("Vector Add on CPU launch\n");

    // Time the kernel
    typedef std::chrono::high_resolution_clock Clock;
    auto tStart = Clock::now();

    vectorAdd(h_A, h_B, h_C, numElements);

    auto duration = Clock::now() - tStart;
    printf("Vector Add on CPU time: %s\n", 
        nanoToString(std::chrono::nanoseconds(duration).count()));

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

