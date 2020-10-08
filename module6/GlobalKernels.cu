#include "KernelFunctionDefinitions.h"

__global__ void globalMemAdd(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    output[tid] = input1[tid] + input2[tid];
}

__global__ void globalMemSub(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    output[tid] = input1[tid] - input2[tid];
}

__global__ void globalMemMult(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    output[tid] = input1[tid] * input2[tid];
}

__global__ void globalMemMod(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    output[tid] = input1[tid] % input2[tid];
}
