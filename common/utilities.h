#ifndef _UTILITIES_H
#define _UTILITIES_H

// Since FLT_EPSILON did not work
// [0][0] 1.32064784 / 1.32064772 {gpu_answer / cpu_answer}
#define CUDA_FLT_EPSILON 1e-5

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Prototypes of utility functions
float randomFloat();
double randomDouble();
int randomInt(int, int);
int randomInt(int);

#endif // _UTILITIES_H
