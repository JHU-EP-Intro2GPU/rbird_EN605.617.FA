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






__global__ void registerMemAdd_4(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 += input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 += input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 += input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 += input2[index4];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }
}

__global__ void registerMemSub_4(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 -= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 -= input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 -= input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 -= input2[index4];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

}

__global__ void registerMemMult_4(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 *= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 *= input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 *= input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 *= input2[index4];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

}

__global__ void registerMemMod_4(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 %= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 %= input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 %= input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 %= input2[index4];
    }

    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

}








__global__ void registerMemAdd_8(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4, value5, value6, value7, value8;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;
    const int index5 = index4 + totalThreads;
    const int index6 = index5 + totalThreads;
    const int index7 = index6 + totalThreads;
    const int index8 = index7 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 += input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 += input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 += input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 += input2[index4];
    }

    if (index5 < count) {
        value5 = input1[index5];
        value5 += input2[index5];
    }

    if (index6 < count) {
        value6 = input1[index6];
        value6 += input2[index6];
    }

    if (index7 < count) {
        value7 = input1[index7];
        value7 += input2[index7];
    }

    if (index8 < count) {
        value8 = input1[index8];
        value8 += input2[index8];
    }


    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

    if (index5 < count) {
        output[index5] = value5;
    }

    if (index6 < count) {
        output[index6] = value6;
    }

    if (index7 < count) {
        output[index7] = value7;
    }

    if (index8 < count) {
        output[index8] = value8;
    }
}

__global__ void registerMemSub_8(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4, value5, value6, value7, value8;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;
    const int index5 = index4 + totalThreads;
    const int index6 = index5 + totalThreads;
    const int index7 = index6 + totalThreads;
    const int index8 = index7 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 -= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 -= input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 -= input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 -= input2[index4];
    }

    if (index5 < count) {
        value5 = input1[index5];
        value5 -= input2[index5];
    }

    if (index6 < count) {
        value6 = input1[index6];
        value6 -= input2[index6];
    }

    if (index7 < count) {
        value7 = input1[index7];
        value7 -= input2[index7];
    }

    if (index8 < count) {
        value8 = input1[index8];
        value8 -= input2[index8];
    }


    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

    if (index5 < count) {
        output[index5] = value5;
    }

    if (index6 < count) {
        output[index6] = value6;
    }

    if (index7 < count) {
        output[index7] = value7;
    }

    if (index8 < count) {
        output[index8] = value8;
    }
}

__global__ void registerMemMult_8(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4, value5, value6, value7, value8;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;
    const int index5 = index4 + totalThreads;
    const int index6 = index5 + totalThreads;
    const int index7 = index6 + totalThreads;
    const int index8 = index7 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 *= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 *= input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 *= input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 *= input2[index4];
    }

    if (index5 < count) {
        value5 = input1[index5];
        value5 *= input2[index5];
    }

    if (index6 < count) {
        value6 = input1[index6];
        value6 *= input2[index6];
    }

    if (index7 < count) {
        value7 = input1[index7];
        value7 *= input2[index7];
    }

    if (index8 < count) {
        value8 = input1[index8];
        value8 *= input2[index8];
    }


    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

    if (index5 < count) {
        output[index5] = value5;
    }

    if (index6 < count) {
        output[index6] = value6;
    }

    if (index7 < count) {
        output[index7] = value7;
    }

    if (index8 < count) {
        output[index8] = value8;
    }
}

__global__ void registerMemMod_8(int* output, const int* input1, const int* input2, const size_t count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int totalThreads = blockDim.x * gridDim.x;

    int value1, value2, value3, value4, value5, value6, value7, value8;

    const int index1 = tid;
    const int index2 = index1 + totalThreads;
    const int index3 = index2 + totalThreads;
    const int index4 = index3 + totalThreads;
    const int index5 = index4 + totalThreads;
    const int index6 = index5 + totalThreads;
    const int index7 = index6 + totalThreads;
    const int index8 = index7 + totalThreads;

    // Load Values
    if (index1 < count) {
        value1 = input1[index1];
        value1 %= input2[index1];
    }

    if (index2 < count) {
        value2 = input1[index2];
        value2 %= input2[index2];
    }

    if (index3 < count) {
        value3 = input1[index3];
        value3 %= input2[index3];
    }

    if (index4 < count) {
        value4 = input1[index4];
        value4 %= input2[index4];
    }

    if (index5 < count) {
        value5 = input1[index5];
        value5 %= input2[index5];
    }

    if (index6 < count) {
        value6 = input1[index6];
        value6 %= input2[index6];
    }

    if (index7 < count) {
        value7 = input1[index7];
        value7 %= input2[index7];
    }

    if (index8 < count) {
        value8 = input1[index8];
        value8 %= input2[index8];
    }


    // Write values
    if (index1 < count) {
        output[index1] = value1;
    }

    if (index2 < count) {
        output[index2] = value2;
    }

    if (index3 < count) {
        output[index3] = value3;
    }

    if (index4 < count) {
        output[index4] = value4;
    }

    if (index5 < count) {
        output[index5] = value5;
    }

    if (index6 < count) {
        output[index6] = value6;
    }

    if (index7 < count) {
        output[index7] = value7;
    }

    if (index8 < count) {
        output[index8] = value8;
    }
}


