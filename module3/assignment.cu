#include <stdio.h>


__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void subtractKernel(int* c, const int* a, const int* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] - b[i];
}

__global__ void multiplyKernel(int* c, const int* a, const int* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}

__global__ void modulusKernel(int* c, const int* a, const int* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] % b[i];
}
