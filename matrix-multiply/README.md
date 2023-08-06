
## Results on an Ubuntu 17.04 CUDA 9.0 Nvidia GeForce GT 650M (Macbook Pro 2012)

```
% ./matrixMultiplyGPU
[Matrix Multiplication of 1024x1024 elements]
Matrix Multiply on GPU launch
Matrix Multiply on GPU time: 805.643ms
Verified for 100 random elements
% sudo /usr/local/cuda-9.0/bin/nvprof --print-gpu-trace ./matrixMultiplyGPU
[Matrix Multiplication of 1024x1024 elements]
==30571== NVPROF is profiling process 30571, command: ./matrixMultiplyGPU
Matrix Multiply on GPU launch
Matrix Multiply on GPU time: 802.415ms
Verified for 100 random elements
==30571== Profiling application: ./matrixMultiplyGPU
==30571== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
177.73ms  700.95us                    -               -         -         -         -  4.0000MB  5.5728GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
178.60ms  693.04us                    -               -         -         -         -  4.0000MB  5.6364GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
179.31ms  799.89ms            (64 64 1)       (16 16 1)        32        0B        0B         -           -           -           -  GeForce GT 650M         1         7  matrixMultiply(float const *, float const *, float*, int) [112]
979.20ms  649.01us                    -               -         -         -         -  4.0000MB  6.0188GB/s      Device    Pageable  GeForce GT 650M         1         7  [CUDA memcpy DtoH]
```

```
% ./matrixMultiplyGPUShared 
[Matrix Multiplication of 1024x1024 elements]
Matrix Multiply Shared on GPU launch
Matrix Multiply Shared on GPU time: 228.631ms
Verified for 100 random elements
% sudo /usr/local/cuda-9.0/bin/nvprof --print-gpu-trace ./matrixMultiplyGPUShared
[Matrix Multiplication of 1024x1024 elements]
==30603== NVPROF is profiling process 30603, command: ./matrixMultiplyGPUShared
Matrix Multiply Shared on GPU launch
Matrix Multiply Shared on GPU time: 226.76ms
Verified for 100 random elements
==30603== Profiling application: ./matrixMultiplyGPUShared
==30603== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
181.81ms  706.55us                    -               -         -         -         -  4.0000MB  5.5286GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
182.67ms  693.24us                    -               -         -         -         -  4.0000MB  5.6348GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
183.37ms  223.56ms            (64 64 1)       (16 16 1)        24  2.0000KB        0B         -           -           -           -  GeForce GT 650M         1         7  matrixMultiply(float const *, float const *, float*, int) [112]
406.93ms  650.26us                    -               -         -         -         -  4.0000MB  6.0072GB/s      Device    Pageable  GeForce GT 650M         1         7  [CUDA memcpy DtoH]
```

```
% ./matrixMultiplyCPU
[Matrix Multiplication of 1024x1024 elements]
Matrix Multiply on CPU launch
Matrix Multiply on CPU time: 794.948ms
Verified for 100 random elements
```
