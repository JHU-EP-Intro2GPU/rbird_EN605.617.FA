#include "KernelFunctionDefinitions.h"

__global__ void registerMemAdd(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int value = input1[tid];
    value += input2[tid];
    output[tid] = value;
}

__global__ void registerMemSub(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int value = input1[tid];
    value -= input2[tid];
    output[tid] = value;
}

__global__ void registerMemMult(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int value = input1[tid];
    value *= input2[tid];
    output[tid] = value;
}

__global__ void registerMemMod(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int value = input1[tid];
    value %= input2[tid];
    output[tid] = value;
}