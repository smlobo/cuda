
## Results on an Ubuntu 17.04 CUDA 9.0 Nvidia GeForce GT 650M (Macbook Pro 2012)

```
% ./vectorAddCuda
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA vectorAdd kernel launch with 196 blocks of 256 threads
Cuda vectorAdd<<<>>> time: 21004 ns
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

```
% ./vectorAddVanilla
[Vector addition of 50000 elements]
Vanilla vectorAdd launch
Vanilla vectorAdd() time: 231305 ns
Test PASSED
Done
```
