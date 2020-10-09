#include "KernelFunctionDefinitions.h"

__global__ void sharedMemAdd(int* output, const int* input1, const int* input2, const size_t count)
{
    extern __shared__ int sharedMem[];

    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        // load both values into sharedMem
        sharedMem[threadIdx.x] = input1[tid];
        sharedMem[threadIdx.x + blockDim.x] = input2[tid];

        int result = sharedMem[threadIdx.x] + sharedMem[threadIdx.x + blockDim.x];
        output[tid] = result;
    }
}

__global__ void sharedMemSub(int* output, const int* input1, const int* input2, const size_t count)
{
    extern __shared__ int sharedMem[];

    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        // load both values into sharedMem
        sharedMem[threadIdx.x] = input1[tid];
        sharedMem[threadIdx.x + blockDim.x] = input2[tid];

        int result = sharedMem[threadIdx.x] - sharedMem[threadIdx.x + blockDim.x];
        output[tid] = result;
    }
}

__global__ void sharedMemMult(int* output, const int* input1, const int* input2, const size_t count)
{
    extern __shared__ int sharedMem[];

    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        // load both values into sharedMem
        sharedMem[threadIdx.x] = input1[tid];
        sharedMem[threadIdx.x + blockDim.x] = input2[tid];

        int result = sharedMem[threadIdx.x] * sharedMem[threadIdx.x + blockDim.x];
        output[tid] = result;
    }
}

__global__ void sharedMemMod(int* output, const int* input1, const int* input2, const size_t count)
{
    extern __shared__ int sharedMem[];

    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        // load both values into sharedMem
        sharedMem[threadIdx.x] = input1[tid];
        sharedMem[threadIdx.x + blockDim.x] = input2[tid];

        int result = sharedMem[threadIdx.x] % sharedMem[threadIdx.x + blockDim.x];
        output[tid] = result;
    }
}
