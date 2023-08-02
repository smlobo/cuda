#include <cstdio>

__global__ void kernel1d(float* dummy) {
    printf("[{%d}] [{%d}] gridDim.x: %d, blockDim.x: %d\n", blockIdx.x, 
        threadIdx.x, gridDim.x, blockDim.x);
    if (threadIdx.x < 1)
        dummy[threadIdx.x] += 1.0;
}

__global__ void kernel2d(float* dummy) {
    printf("[{%d,%d}] [{%d,%d}] gridDim.x: %d, gridDim.y: %d | blockDim.x: %d, "
        "blockDim.y: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, 
        gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    if (threadIdx.x < 1)
        dummy[threadIdx.x] += 1.0;
}

int main(void) {
    dim3 block, thread;

    void* dummyPtr;
    cudaMalloc(&dummyPtr, 2*sizeof(float));

    block.x = 3;
    thread.x = 2;
    printf("Launching 1d kernel with <<<%d,%d>>>\n", block.x, thread.x);
    kernel1d<<<block,thread>>>((float*)dummyPtr);

    block.x = 1;
    block.y = 2;
    thread.x = 3;
    thread.y = 4;
    printf("Launching 2d kernel with <<<{%d;%d},{%d;%d}>>>\n", block.x, block.y, 
        thread.x, thread.y);
    kernel2d<<<block,thread>>>((float*)dummyPtr);

    cudaFree(dummyPtr);

    return 0;
}

