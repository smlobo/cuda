#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "utilities.h"

void verify(const float *A, const float *B, const float *C, int numElements, 
    int p, int q, int count) {

    // Verify that the result for C[p][q] is accurate
    float sum = 0.0;
    for (int i = 0; i < numElements; ++i)
        sum += A[p*numElements+i] * B[i*numElements+q];

    int index = p*numElements + q;
    if (fabsf(C[index] - sum) > CUDA_FLT_EPSILON) {
        printf("Verification FAIL <%d> [%d][%d] %.4f / %.4f\n", count, p, q, 
            C[index], sum);
        exit(EXIT_FAILURE);
    }
}
