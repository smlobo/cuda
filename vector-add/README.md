
## Results on an Ubuntu 17.04 CUDA 9.0 Nvidia GeForce GT 650M (Macbook Pro 2012)

```
% ./vectorAddGPU
[Vector addition of 10000000 elements]
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 39063 blocks of 256 threads
Copy output data from the CUDA device to the host memory
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
