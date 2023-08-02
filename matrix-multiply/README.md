
## Results on an Ubuntu 17.04 CUDA 9.0 Nvidia GeForce GT 650M (Macbook Pro 2012)

```
% ./matrixMultiplyGPU
[Matrix Multiplication of 1024x1024 elements]
Matrix Multiply on GPU launch
Matrix Multiply on GPU time (kernel): 22253 ns
Matrix Multiply on GPU time (with copy): 1021751067 ns
Verified for 100 random elements
```

```
% ./matrixMultiplyGPUShared 
[Matrix Multiplication of 1024x1024 elements]
Matrix Multiply Shared on GPU launch
Matrix Multiply Shared on GPU time (kernel): 21865 ns
Matrix Multiply Shared on GPU time (with copy): 436495909 ns
Verified for 100 random elements
```

```
% ./matrixMultiplyCPU
./matrixMultiplyCPU 
[Matrix Multiplication of 1024x1024 elements]
Matrix Multiply on CPU launch
Matrix Multiply on CPU time: 15491845464 ns
Verified for 100 random elements
```
