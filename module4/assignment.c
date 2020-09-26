
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "assignment.h"

#include <stdio.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <time.h>

void populateTestData(int threadCount, int blocksize);
void printVector(const HostMemory<int>&, size_t countToUse);
void performCalculations(int blocksize);
void runVerification();

#pragma region Cuda Math Kernels
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

#pragma endregion

enum KernalToRun {
    RunAddKernel, RunSubtractKernel, RunMultiplyKernel, RunModulusKernel
};

HostMemory<int> firstSourceArray;
HostMemory<int> secondSourceArray;

HostMemory<int> addResults;
HostMemory<int> subtractResults;
HostMemory<int> multiplyResults;
HostMemory<int> modulusResults;

bool printDebug = false;
bool verifyCorrectness = true;

int main(int argc, char* argv[])
{
    int totalThreadCount = 20;
    int blockSize = 20; // number of threads per block

    // parse the command line arguments
    for (int i = 0; i < argc; i++) {
        const char* arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            printf("USAGE: ./assignment.exe [num_threads] [blocksize]");
            printf("Optional Arguments:\n");
            printf("--help, -h: Show this message and exit\n\n");
            printf("--blocksize [block_size]: set the block size to use on the cuda kernels. Default to %d\n", blockSize);
            printf("--threads [thread_count]: set the thread count (and test array size) to use on the cuda kernels. Default to %d\n", totalThreadCount);
            printf("--debug: print out the data to determine test input and output\n");
            printf("--verify: verify that the results from the gpu are the expected values\n");
            return 0;
        }
        else if (strcmp(arg, "--blocksize") == 0) {
            i++;
            blockSize = atoi(argv[i]);
        }
        else if (strcmp(arg, "--threads") == 0) {
            i++;
            totalThreadCount = atoi(argv[i]);
        }
        else if (strcmp(arg, "--debug") == 0) {
            printDebug = true;
        }
        else if (strcmp(arg, "--verify") == 0) {
            verifyCorrectness = true;
        }
        else if (i == 1) {
            totalThreadCount = atoi(arg);
        }
        else if (i == 2) {
            blockSize = atoi(arg);
        }

    }

    populateTestData(totalThreadCount, blockSize);

    if (printDebug) {
        printf("first array:\n");
        printVector(firstSourceArray, totalThreadCount);
        printf("\n\nsecond array:\n");
        printVector(secondSourceArray, totalThreadCount);
        printf("\n\n");
    }

    performCalculations(blockSize);

    if (printDebug) {
        for (int i = 0; i < totalThreadCount; i++) {
            int a = firstSourceArray.ptr()[i];
            int b = secondSourceArray.ptr()[i];

            printf("%3d: %2d + %d = %3d    ", i, a, b, addResults.ptr()[i]);
            printf("%2d - %d = %3d    ", a, b, subtractResults.ptr()[i]);
            printf("%2d * %d = %3d    ", a, b, multiplyResults.ptr()[i]);
            printf("%2d %% %d = %3d    ", a, b, modulusResults.ptr()[i]);
            printf("\n");
        }
    }

    if (verifyCorrectness) {
        runVerification();
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    gpuErrchk(cudaDeviceReset());

    return 0;
}

int* createDeviceBuffer(size_t bytes) {
    int* devicePtr = nullptr;
    gpuErrchk(cudaMalloc((void**)&devicePtr, bytes));
    return devicePtr;
}

// Use helper function tomake performCalculations more readable
void testKernel(KernalToRun runKernel, const int numBlocks, const int blocksize, int* device_destPtr, int* device_Source1, int* device_Source2);

void performCalculations(int blocksize)
{
    cudaError_t cudaStatus;

    size_t bufferCount = firstSourceArray.size();
    const size_t totalBytes = bufferCount * sizeof(int);
    const int numBlocks = bufferCount / blocksize;

    // input sources
    DeviceMemory<int> dev_firstSource(bufferCount);
    DeviceMemory<int> dev_secondSource(bufferCount);

    // output sources
    DeviceMemory<int> dev_addResults(bufferCount);
    DeviceMemory<int> dev_subtractResults(bufferCount);
    DeviceMemory<int> dev_multiplyResults(bufferCount);
    DeviceMemory<int> dev_modulusResults(bufferCount);

    // Copy input vectors from host memory to GPU buffers.
    gpuErrchk(cudaMemcpy(dev_firstSource.ptr(), firstSourceArray.ptr(), totalBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_secondSource.ptr(), secondSourceArray.ptr(), totalBytes, cudaMemcpyHostToDevice));

    // Wait for kernels to finish so that we can properly time each kernel run
    gpuErrchk(cudaDeviceSynchronize());

    // Launch and time each AddKernel (helper function synchronizes with device and prints out time)
    printf("Running AddKernel\n");
    testKernel(KernalToRun::RunAddKernel, numBlocks, blocksize, dev_addResults.ptr(), dev_firstSource.ptr(), dev_secondSource.ptr());

    printf("Running SubtractKernel\n");
    testKernel(KernalToRun::RunSubtractKernel, numBlocks, blocksize, dev_subtractResults.ptr(), dev_firstSource.ptr(), dev_secondSource.ptr());

    printf("Running MultiplyKernel\n");
    testKernel(KernalToRun::RunMultiplyKernel, numBlocks, blocksize, dev_multiplyResults.ptr(), dev_firstSource.ptr(), dev_secondSource.ptr());

    printf("Running ModulusKernel\n");
    testKernel(KernalToRun::RunModulusKernel, numBlocks, blocksize, dev_modulusResults.ptr(), dev_firstSource.ptr(), dev_secondSource.ptr());


    // Copy output vectors from GPU buffer to host memory.
    gpuErrchk(cudaMemcpy(addResults.ptr(), dev_addResults.ptr(), totalBytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(subtractResults.ptr(), dev_subtractResults.ptr(), totalBytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(multiplyResults.ptr(), dev_multiplyResults.ptr(), totalBytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(modulusResults.ptr(), dev_modulusResults.ptr(), totalBytes, cudaMemcpyDeviceToHost));
}

void testKernel(KernalToRun runKernel, const int numBlocks, const int blocksize, int* device_destPtr, int* device_Source1, int* device_Source2) {
    if (printDebug) {
        printf("\tLaunching %d blocks with %d threads per block.\n", numBlocks, blocksize);
    }

    TimeCodeBlock timeKernelRun("\tKernelRun");

    switch (runKernel)
    {
    case RunAddKernel:
        addKernel << < numBlocks, blocksize >> > (device_destPtr, device_Source1, device_Source2);
        break;
    case RunSubtractKernel:
        subtractKernel << < numBlocks, blocksize >> > (device_destPtr, device_Source1, device_Source2);
        break;
    case RunMultiplyKernel:
        multiplyKernel << < numBlocks, blocksize >> > (device_destPtr, device_Source1, device_Source2);
        break;
    case RunModulusKernel:
        modulusKernel << < numBlocks, blocksize >> > (device_destPtr, device_Source1, device_Source2);
        break;
    default:
        break;
    }

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void populateTestData(const int threadCount, const int blocksize)
{
    // Reserve enough data so that we don't have any out of bounds
    // memory access (prevent the need to check device array size).
    // We need a number >= threadcount that is evenly divisible by blocksize
    size_t reserveSize = 0;
    const int extraThreads = threadCount % blocksize;
    if (extraThreads == 0)
        reserveSize = threadCount; // blocks fit exactly
    else
        reserveSize = threadCount + (blocksize - extraThreads); // add more threads to be a multiple of 'blocksize'

    {
        TimeCodeBlock hostAllocation("Allocate Host Memory");
        firstSourceArray.allocate(reserveSize);
        secondSourceArray.allocate(reserveSize);
    }

    {
        TimeCodeBlock deviceAllocation("Device Allocation");
        addResults.allocate(reserveSize);
        subtractResults.allocate(reserveSize);
        multiplyResults.allocate(reserveSize);
        modulusResults.allocate(reserveSize);
    }

    {
        TimeCodeBlock populateHostData("Populate host data");
        // Populate the first array: "the first should contain values from 0 - total number of threads"
        for (int i = 0; i < threadCount; i++)
            firstSourceArray.ptr()[i] = i;

        // the second with random values between 0 and 3
        int maxValueExclusive = 4;

        // randomize the seed
        srand(time(NULL));
        for (int i = 0; i < threadCount; i++)
            secondSourceArray.ptr()[i] = rand() % maxValueExclusive;
    }
}

void printVector(const HostMemory<int>& values, size_t countToUse) {
    printf("[");

    const int* data = values.ptr();
    for (size_t i = 0; i < countToUse; i++) {
        if (i != 0) {
            printf(",");
        }

        printf(" %d", data[i]);
    }

    printf("]");
}

void runVerification() {
    int errorCount = 0;

    for (size_t i = 0; i < firstSourceArray.size(); i++) {
        int a = firstSourceArray.ptr()[i];
        int b = secondSourceArray.ptr()[i];

        if (a + b != addResults.ptr()[i]) {
            printf("ERROR: %d Add is incorrect\n", i);
            errorCount++;
        }

        if (a - b != subtractResults.ptr()[i]) {
            printf("ERROR: %d Subtract is incorrect\n", i);
            errorCount++;
        }

        if (a * b != multiplyResults.ptr()[i]) {
            printf("ERROR: %d Multiply is incorrect\n", i);
            errorCount++;
        }

        if (b == 0) {
            if (modulusResults.ptr()[i] != -1) {
                printf("ERROR: %d Modulus is incorrect\n", i);
                errorCount++;
            }
        }
        else if (a % b != modulusResults.ptr()[i]) {
            printf("ERROR: %d Modulus is incorrect\n", i);
            errorCount++;
        }

    }

    if (errorCount == 0) {
        printf("Verification Success!\n");
    }

}
