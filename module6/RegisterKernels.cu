#include "KernelFunctionDefinitions.h"

__global__ void registerMemAdd(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        int value = input1[tid];
        value += input2[tid];
        output[tid] = value;
    }
}

__global__ void registerMemSub(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        int value = input1[tid];
        value -= input2[tid];
        output[tid] = value;
    }
}

__global__ void registerMemMult(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        int value = input1[tid];
        value *= input2[tid];
        output[tid] = value;
    }
}

__global__ void registerMemMod(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < count) {
        int value = input1[tid];
        value %= input2[tid];
        output[tid] = value;
    }
}


__global__ void registerMemAdd_2(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2;
    
    const int index1 = tid;
    const int index2 = index1 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 += input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 += input2[index2];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }
}

__global__ void registerMemSub_2(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 -= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 -= input2[index2];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }
}

__global__ void registerMemMult_2(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 *= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 *= input2[index2];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }
}

__global__ void registerMemMod_2(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 %= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 %= input2[index2];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }
}


