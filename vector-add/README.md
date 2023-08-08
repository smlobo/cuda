
## Results on an Ubuntu 22.04 CUDA 12.2 Nvidia Quadro T2000 (Lenovo ThinkPad P53)

```
% ./vectorAddGPU
[Vector addition of 1M elements]
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 3907 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Vector Add on GPU time: 3.743ms
Test PASSED
% nvprof --print-gpu-trace ./vectorAddGPU
[Vector addition of 1M elements]
==33973== NVPROF is profiling process 33973, command: ./vectorAddGPU
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 3907 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Vector Add on GPU time: 3.962ms
Test PASSED
==33973== Profiling application: ./vectorAddGPU
==33973== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
3.16810s  690.72us                    -               -         -         -         -  3.8147MB  5.3934GB/s    Pageable      Device  Quadro T2000 (0         1         7  [CUDA memcpy HtoD]
3.16895s  618.56us                    -               -         -         -         -  3.8147MB  6.0225GB/s    Pageable      Device  Quadro T2000 (0         1         7  [CUDA memcpy HtoD]
3.16965s  103.20us           (3907 1 1)       (256 1 1)        16        0B        0B         -           -           -           -  Quadro T2000 (0         1         7  vectorAdd(float const *, float const *, float*, int) [130]
3.16977s  1.2098ms                    -               -         -         -         -  3.8147MB  3.0794GB/s      Device    Pageable  Quadro T2000 (0         1         7  [CUDA memcpy DtoH]
```

```
% ./vectorAddGPUCublas 
[Vector addition of 1M elements]
Vector Add on GPU with libcublas time: 3.900ms
Test PASSED
% nvprof --print-gpu-trace ./vectorAddGPUCublas
[Vector addition of 1M elements]
==34016== NVPROF is profiling process 34016, command: ./vectorAddGPUCublas
Vector Add on GPU with libcublas time: 4.100ms
Test PASSED
==34016== Profiling application: ./vectorAddGPUCublas
==34016== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
3.13534s  727.81us                    -               -         -         -         -  3.8147MB  5.1185GB/s    Pageable      Device  Quadro T2000 (0         1        13  [CUDA memcpy HtoD]
3.13612s  601.09us                    -               -         -         -         -  3.8147MB  6.1976GB/s    Pageable      Device  Quadro T2000 (0         1        13  [CUDA memcpy HtoD]
3.13716s  103.65us           (3907 1 1)       (256 1 1)        32        0B        0B         -           -           -           -  Quadro T2000 (0         1        13  void axpy_kernel_val<float, float>(cublasAxpyParamsVal<float, float, float>) [1184]
3.13726s  1.1934ms                    -               -         -         -         -  3.8147MB  3.1216GB/s      Device    Pageable  Quadro T2000 (0         1        13  [CUDA memcpy DtoH]
```

## Results on an Ubuntu 17.04 CUDA 9.0 Nvidia GeForce GT 650M (Macbook Pro 2012)

```
% ./vectorAddGPU
[Vector addition of 10000000 elements]
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 39063 blocks of 256 threads
Copy output data from the CUDA devi```
ce to the host memory
Vector Add on GPU time: 31.708ms
Test PASSED
% sudo /usr/local/cuda-9.0/bin/nvprof --print-gpu-trace ./vectorAddGPU
[Vector addition of 10000000 elements]
==30176== NVPROF is profiling process 30176, command: ./vectorAddGPU
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 39063 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Vector Add on GPU time: 31.868ms
Test PASSED
==30176== Profiling application: ./vectorAddGPU
==30176== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
182.96ms  6.4049ms                    -               -         -         -         -  38.147MB  5.8163GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
189.49ms  6.3746ms                    -               -         -         -         -  38.147MB  5.8440GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
195.89ms  3.4072ms          (39063 1 1)       (256 1 1)         8        0B        0B         -           -           -           -  GeForce GT 650M         1         7  vectorAdd(float const *, float const *, float*, int) [112]
199.30ms  14.872ms                    -               -         -         -         -  38.147MB  2.5049GB/s      Device    Pageable  GeForce GT 650M         1         7  [CUDA memcpy DtoH]
```

```
% ./vectorAddGPUGridStride 
[Vector addition of 10000000 elements]
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 1024 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Vector Add with Grid Stride on GPU time: 31.415ms
Test PASSED
% sudo /usr/local/cuda-9.0/bin/nvprof --print-gpu-trace ./vectorAddGPUGridStride
[Vector addition of 10000000 elements]
==11739== NVPROF is profiling process 11739, command: ./vectorAddGPUGridStride
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 1024 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Vector Add with Grid Stride on GPU time: 31.307ms
Test PASSED
==11739== Profiling application: ./vectorAddGPUGridStride
==11739== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
180.07ms  6.3979ms                    -               -         -         -         -  38.147MB  5.8227GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
186.60ms  6.3875ms                    -               -         -         -         -  38.147MB  5.8322GB/s    Pageable      Device  GeForce GT 650M         1         7  [CUDA memcpy HtoD]
193.00ms  2.6595ms           (1024 1 1)       (256 1 1)        10        0B        0B         -           -           -           -  GeForce GT 650M         1         7  vectorAdd(float const *, float const *, float*, int) [112]
195.66ms  15.065ms                    -               -         -         -         -  38.147MB  2.4728GB/s      Device    Pageable  GeForce GT 650M         1         7  [CUDA memcpy DtoH]
```

```
% ./vectorAddCPU 
[Vector addition of 10000000 elements]
Vector Add on CPU launch
Vector Add on CPU time: 15.445ms
Test PASSED
```
