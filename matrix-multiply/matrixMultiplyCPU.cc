/*
 * Matrix Multiplication: C = A x B.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cassert>
#include <cfloat>

#include "utilities.h"

const int numVerifications = 100;

void verify(const float*, const float*, const float*, int, int, int, int);

void matrixMultiply(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        for (int j = 0; j < numElements; j++) {
            float sum = 0.0;
            for (int k = 0; k < numElements; k++)
                sum += A[i*numElements+k] * B[k*numElements+j];
            C[i*numElements+j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    // Print the vector length to be used, and compute its size
    int numElements = 1024;
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

    // Launch matrix multiply on the CPU
    printf("Matrix Multiply on CPU launch\n");

    // Time the matrix multiply
    typedef std::chrono::high_resolution_clock Clock;
    auto tStart = Clock::now();

    matrixMultiply(h_A, h_B, h_C, numElements);

    auto duration = Clock::now() - tStart;
    printf("Matrix Multiply on CPU time: %ld ns\n", 
        std::chrono::nanoseconds(duration).count());

    for (int i = 0; i < numVerifications; i++) {
        int p = randomInt(numElements);
        int q = randomInt(numElements);
        verify(h_A, h_B, h_C, numElements, p, q, i);
    }
    printf("Verified for %d random elements\n", numVerifications);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
