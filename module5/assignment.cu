
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "assignment.h"

#include <cstring>
#include <cstdlib>

#include <stdio.h>

// For simplicity in the shared memory kernels, there must be an exact fit of blockSize in the array:
// arraySize % blockSize == 0
//const int arraySize = 512;
//const int blockSize = 128;

//#define arraySize 10
//#define blockSize 10

//#define arraySize 15000
//#define blockSize 500


//#define blockSize 256
// vocareum tests at blockSize * 50000
//#define arraySize (blockSize * 250000)


#define blockSize 256
#define arraySize (blockSize * 25000)


#define ITERATIONS 5000


const int numBlocks = arraySize / blockSize;

static_assert(arraySize % blockSize == 0, "This program only supports array sizes that fit the block size exactly.");

// Use static global memory variables to learn how to use them (would use cudaMalloc otherwise)
__device__ static int gmem_input[arraySize];
__device__ static int gmem_output[arraySize];

__device__ static int gmem_shift_value;

#define MAX_CONST_ARRAY_SIZE 15000

#if arraySize <= MAX_CONST_ARRAY_SIZE
// const memory is limited
__constant__ int const_input[arraySize];
#endif


__constant__ int const_shift_value;

__constant__ int const_value_1;
__constant__ int const_value_2;
__constant__ int const_value_3;

const int shift_value_for_const_test = 3;

const int value1_for_const_test = 208;
const int value2_for_const_test = 517;
const int value3_for_const_test = 28;


// Host buffers
static int host_input[arraySize];
static int host_output[arraySize];


// Math Problem:
// output[i] = ((input[i - 1] + input[i]) * input[i + 1]) >> const_shift_val

// boundaries wrap around: 0 - 1 -> access memory input[N - 1]
//                         N-1 + 1 -> access memory input[0] 

// This math problem does not have any special meaning. I went for something interesting.


enum TestKernelType {
    GLOBAL_MEM, SHARED_MEM, CONST_MEM,
    GLOBAL_MEM_WITH_PARAM, SHARED_MEM_WITH_PARAM, CONST_MEM_ARRAY
};

#pragma region CUDA Kernels
// Kernels will simply assume that host will set up boundaries correctly in order to
// simplify kernel code. For a size 'N' input/output, there will be exactly N threads launched.

__device__ void kernelMathFunctionGlobalMemory(const int constant_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int lowerIndex = i - 1;
    if (lowerIndex < 0)
        lowerIndex = arraySize - 1;

    int upperIndex = (i + 1) % arraySize;

    // run multiple iterations simply to stress the memory. Calculation is the same as 1 iteration
    int value = 0;
    for (int count = 0; count < ITERATIONS; count++) {
        value = gmem_input[lowerIndex] + gmem_input[i];
        value *= gmem_input[upperIndex];
        value >>= constant_value;
    }

    gmem_output[i] = value;
}

// constant value will be passed as a function parameter
__global__ void globalMemoryKernelWithConstantParameter(const int constant_value)
{
    kernelMathFunctionGlobalMemory(constant_value);
}

// constant_value will reside in global memory
__global__ void globalMemoryKernel()
{
    kernelMathFunctionGlobalMemory(gmem_shift_value);
}


__device__ void kernelMathFunctionSharedMemory(int* shared_memory, const int constant_value)
{
    // shared index cannot accept a negative index for thread 0. To allow for this, increase the pointer by 1 location
    // for all threads
    shared_memory = shared_memory + 1;
    const int sharedMemoryIndex = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadIndex = threadIdx.x;

    // load global memory into shared memory. Account for i - 1 < 0
    int lowerIndex = i - 1;
    if (lowerIndex < 0)
        lowerIndex = arraySize - 1;

    shared_memory[threadIndex - 1] = gmem_input[lowerIndex];

    // load the last values in the block. The last thread's [i] value and [i + 1] value
    if (threadIndex + 2 >= blockDim.x) {
        // load i + 1. Account for i + 1 == arraySize
        int upperIndex = (i + 1) % arraySize;
        shared_memory[threadIndex + 1] = gmem_input[upperIndex];
    }

    __syncthreads();

    // run multiple iterations simply to stress the memory. Calculation is the same as 1 iteration
    int value = 0;
    for (int count = 0; count < ITERATIONS; count++) {
        value = shared_memory[sharedMemoryIndex - 1] + shared_memory[sharedMemoryIndex];
        value *= shared_memory[sharedMemoryIndex + 1];
        value >>= constant_value;
    }

    gmem_output[i] = value;
}


// constant value will be passed as a function parameter
__global__ void sharedMemoryKernelWithConstantParameter(const int constant_value)
{
    // extra shared memory: index -1, index for last thread + 1
    __shared__ int shared_memory[blockSize + 2];
    kernelMathFunctionSharedMemory(shared_memory, constant_value);
}

// constant_value will reside in global memory
__global__ void sharedMemoryKernel()
{
    // extra shared memory: index -1, index for last thread + 1
    __shared__ int shared_memory[blockSize + 2];

    // load global constant into shared memory
    if (threadIdx.x == 0) {
        shared_memory[0] = gmem_shift_value;
    }

    __syncthreads();

    // load shared memory into local memory, sync threads before overwriting index 0
    const int local_shift_value = shared_memory[0];

    __syncthreads();

    kernelMathFunctionSharedMemory(shared_memory, local_shift_value);
}

__global__ void constMemoryKernel() {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // run multiple iterations simply to stress the memory. Calculation is the same as 1 iteration
    int value = 0;
    for (int count = 0; count < ITERATIONS; count++) {
        value = const_value_1 + const_value_2;
        value *= const_value_3;
        value >>= const_shift_value;
    }

    gmem_output[i] = value;
}

__global__ void constMemoryKernelReadFromArray()
{
#if arraySize <= MAX_CONST_ARRAY_SIZE
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int lowerIndex = i - 1;
    if (lowerIndex < 0)
        lowerIndex = arraySize - 1;

    int upperIndex = (i + 1) % arraySize;
    #
    // run multiple iterations simply to stress the memory. Calculation is the same as 1 iteration
    int value = 0;
    for (int count = 0; count < ITERATIONS; count++) {
        value = const_input[lowerIndex] + const_input[i];
        value *= const_input[upperIndex];
        value >>= const_shift_value;
    }


    gmem_output[i] = value;
#endif
}

#pragma endregion

void populateTestData() {
    for (int i = 0; i < arraySize; i++) {
        host_input[i] = i + 1; // rand() % 
    }

    // send input buffer to device
    gpuErrchk(cudaMemcpyToSymbol(gmem_input, host_input, sizeof(gmem_input)));
}

void validateCorrectness(const int shiftValue, bool isConstMemory=false) {
    gpuErrchk(cudaMemcpyFromSymbol(host_output, gmem_output, sizeof(host_output)));

    int expectedConstResult = value1_for_const_test + value2_for_const_test;
    expectedConstResult *= value3_for_const_test;
    expectedConstResult >>= shift_value_for_const_test;

    for (int i = 0; i < arraySize; i++) {
        int expectedAnswer;
        if (isConstMemory) {
            expectedAnswer = expectedConstResult;
        }
        else {
            int lowerIndex = (i == 0 ? arraySize - 1 : i - 1);
            int upperIndex = (i + 1) % arraySize;

            expectedAnswer = ((host_input[lowerIndex] + host_input[i]) * (host_input[upperIndex]) >> shiftValue);
        }

        //printf("%3d: ((%d + %d) * %d) >> %d) = %d\n", i, host_input[lowerIndex], host_input[i], host_input[upperIndex], shiftValue, expectedAnswer);

        if (host_output[i] != expectedAnswer) {
            printf("%3d: Error! Expected: %3d Actual: %3d\n", i, expectedAnswer, host_output[i]);
        }
    }
}

void resetOutputBufferData() {
    int* d_output = nullptr;
    gpuErrchk(cudaGetSymbolAddress((void**)&d_output, gmem_output));

    // clear output buffers
    gpuErrchk(cudaMemset(d_output, 0, sizeof(gmem_output)));
    memset(host_output, 0, sizeof(host_output));
}

void testKernelRun(TestKernelType kernelType, const int shiftValue, const char* description) {
    cudaMemcpyToSymbol(gmem_shift_value, &shiftValue, sizeof(shiftValue));

    {
        TimeCodeBlock kernelRunMeasurement(description);

        switch (kernelType)
        {
        case GLOBAL_MEM:
            globalMemoryKernel <<<numBlocks, blockSize >>> ();
            break;
        case GLOBAL_MEM_WITH_PARAM:
            globalMemoryKernelWithConstantParameter <<<numBlocks, blockSize >>> (shiftValue);
            break;
        case SHARED_MEM:
            sharedMemoryKernel <<<numBlocks, blockSize >>>();
            break;
        case SHARED_MEM_WITH_PARAM:
            sharedMemoryKernelWithConstantParameter <<<numBlocks, blockSize >>> (shiftValue);
            break;
        case CONST_MEM:
            constMemoryKernel<<<numBlocks, blockSize >>>();
            break;
        case CONST_MEM_ARRAY:
            constMemoryKernelReadFromArray<<<numBlocks, blockSize>>>();
            break;
        default:
            break;
        }

        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    bool validateForConstMemory = kernelType == TestKernelType::CONST_MEM;
    validateCorrectness(shiftValue, validateForConstMemory);
    resetOutputBufferData();
}

void testKernelsLoadingShiftFromGlobalMemory(const int shiftValue)
{
    printf("Test loading the shift value from global memory\n\n");

//    for (const auto& testType : {TestKernelType::GLOBAL_MEM, TestKernelType::SHARED_MEM, TestKernelType::CONST_MEM})


}

void testKernels() {
    printf("Arraysize: %d Blocksize: %d Iterations: %d\n", arraySize, blockSize, ITERATIONS);
    populateTestData();

    resetOutputBufferData();
    
    const int shiftValue = 3;

    printf("--------------- GLOBAL MEMORY TESTS -------------------------\n");

    testKernelRun(TestKernelType::GLOBAL_MEM, shiftValue, "Global Memory Kernel, Global Memory Shift Value");
    testKernelRun(TestKernelType::GLOBAL_MEM_WITH_PARAM, shiftValue, "Global Memory Kernel, Shift Value as Parameter");


    printf("\n--------------- SHARED MEMORY TESTS -------------------------\n");
    testKernelRun(TestKernelType::SHARED_MEM, shiftValue, "Shared Memory Kernel, Global Memory Shift Value");
    testKernelRun(TestKernelType::SHARED_MEM_WITH_PARAM, shiftValue, "Shared Memory Kernel, Shift Value as Parameter");


    cudaMemcpyToSymbol(const_shift_value, &shift_value_for_const_test, sizeof(shiftValue));
    cudaMemcpyToSymbol(const_value_1, &value1_for_const_test, sizeof(shiftValue));
    cudaMemcpyToSymbol(const_value_2, &value2_for_const_test, sizeof(shiftValue));
    cudaMemcpyToSymbol(const_value_3, &value3_for_const_test, sizeof(shiftValue));

    printf("\n--------------- CONST MEMORY TESTS -------------------------\n");
    testKernelRun(TestKernelType::CONST_MEM, shiftValue, "Constant Memory Kernel");

#if arraySize <= MAX_CONST_ARRAY_SIZE
    // const memory is limited
    gpuErrchk(cudaMemcpyToSymbol(const_input, host_input, sizeof(const_input)));
    testKernelRun(TestKernelType::CONST_MEM_ARRAY, shiftValue, "Constant Memory Kernel, Read Const Memory Array");
#endif
}


int main(int argc, char* argv[])
{
    testKernels();

    return 0;
}

