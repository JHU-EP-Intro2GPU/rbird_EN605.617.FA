
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <vector>

// Helper function and #DEFINE to help with error checking (found from stackoverflow)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void populateTestData(int threadCount, int blocksize);
void printVector(const std::vector<int>&, size_t countToUse);
void performCalculations(int blocksize);
void runVerification();

enum KernalToRun {
    RunAddKernel, RunSubtractKernel, RunMultiplyKernel, RunModulusKernel
};

std::vector<int> firstSourceArray;
std::vector<int> secondSourceArray;

std::vector<int> addResults;
std::vector<int> subtractResults;
std::vector<int> multiplyResults;
std::vector<int> modulusResults;

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
        //printf("\nAdd Results:\n");
        //printVector(addResults);
        //printf("\nSubtract Results:\n");
        //printVector(subtractResults);
        //printf("\nMultiply Results:\n");
        //printVector(multiplyResults);
        //printf("\nModulus Results:\n");
        //printVector(modulusResults);
        //printf("\n");

        for (int i = 0; i < totalThreadCount; i++) {
            int a = firstSourceArray[i];
            int b = secondSourceArray[i];

            printf("%3d: %2d + %d = %3d    ", i, a, b, addResults[i]);
            printf("%2d - %d = %3d    ", a, b, subtractResults[i]);
            printf("%2d * %d = %3d    ", a, b, multiplyResults[i]);
            printf("%2d %% %d = %3d    ", a, b, modulusResults[i]);
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

    size_t bufferCount = firstSourceArray.capacity();
    const size_t totalBytes = bufferCount * sizeof(int);
    const int numBlocks = bufferCount / blocksize;

    // input sources
    int* dev_firstSource = createDeviceBuffer(totalBytes);
    int* dev_secondSource = createDeviceBuffer(totalBytes);

    // output sources
    int* dev_addResults = createDeviceBuffer(totalBytes);
    int* dev_subtractResults = createDeviceBuffer(totalBytes);
    int* dev_multiplyResults = createDeviceBuffer(totalBytes);
    int* dev_modulusResults = createDeviceBuffer(totalBytes);

    // Copy input vectors from host memory to GPU buffers.
    int* testBuffer = new int[bufferCount];

    for (int i = 0; i < bufferCount; i++) {
        testBuffer[i] = firstSourceArray[i];
    }


    gpuErrchk(cudaMemcpy(dev_firstSource, testBuffer, totalBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_secondSource, secondSourceArray.data(), totalBytes, cudaMemcpyHostToDevice));

    // Wait for kernels to finish so that we can properly time each kernel run
    gpuErrchk(cudaDeviceSynchronize());

    // Launch and time each AddKernel (helper function synchronizes with device and prints out time)
    printf("Running AddKernel\n");
    testKernel(KernalToRun::RunAddKernel, numBlocks, blocksize, dev_addResults, dev_firstSource, dev_secondSource);

    printf("Running SubtractKernel\n");
    testKernel(KernalToRun::RunSubtractKernel, numBlocks, blocksize, dev_subtractResults, dev_firstSource, dev_secondSource);

    printf("Running MultiplyKernel\n");
    testKernel(KernalToRun::RunMultiplyKernel, numBlocks, blocksize, dev_multiplyResults, dev_firstSource, dev_secondSource);

    printf("Running ModulusKernel\n");
    testKernel(KernalToRun::RunModulusKernel, numBlocks, blocksize, dev_modulusResults, dev_firstSource, dev_secondSource);


    // Copy output vectors from GPU buffer to host memory.
    gpuErrchk(cudaMemcpy(addResults.data(), dev_addResults, totalBytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(subtractResults.data(), dev_subtractResults, totalBytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(multiplyResults.data(), dev_multiplyResults, totalBytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(modulusResults.data(), dev_modulusResults, totalBytes, cudaMemcpyDeviceToHost));

Error:
    cudaFree(dev_firstSource);
    cudaFree(dev_secondSource);

    cudaFree(dev_addResults);
    cudaFree(dev_subtractResults);
    cudaFree(dev_multiplyResults);
    cudaFree(dev_modulusResults);
}

void testKernel(KernalToRun runKernel, const int numBlocks, const int blocksize, int* device_destPtr, int* device_Source1, int* device_Source2) {
    if (printDebug) {
        printf("\tLaunching %d blocks with %d threads per block.\n", numBlocks, blocksize);
    }

    auto startTime = std::chrono::high_resolution_clock::now();

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

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::microseconds timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "\tTime to run kernel: " << timeDiff.count() << " microseconds." << std::endl;
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

    // resize properly works with copying straight into vector from device memory
    firstSourceArray.resize(reserveSize);
    secondSourceArray.resize(reserveSize);

    addResults.resize(reserveSize);
    subtractResults.resize(reserveSize);
    multiplyResults.resize(reserveSize);
    modulusResults.resize(reserveSize);

    // Populate the first array: "the first should contain values from 0 – total number of threads"
    for (int i = 0; i < threadCount; i++)
        firstSourceArray[i] = i;

    // the second with random values between 0 and 3
    int maxValueExclusive = 4;

    // randomize the seed
    srand(time(NULL));
    for (int i = 0; i < threadCount; i++)
        secondSourceArray[i] = rand() % maxValueExclusive;
}

void printVector(const std::vector<int>& values, size_t countToUse) {
    printf("[");

    for (size_t i = 0; i < countToUse; i++) {
        if (i != 0) {
            printf(",");
        }

        printf(" %d", values[i]);
    }

    printf("]");
}

void runVerification() {
    int errorCount = 0;

    for (size_t i = 0; i < firstSourceArray.size(); i++) {
        int a = firstSourceArray[i];
        int b = secondSourceArray[i];

        if (a + b != addResults[i]) {
            printf("ERROR: %d Add is incorrect\n", i);
            errorCount++;
        }

        if (a - b != subtractResults[i]) {
            printf("ERROR: %d Subtract is incorrect\n", i);
            errorCount++;
        }

        if (a * b != multiplyResults[i]) {
            printf("ERROR: %d Multiply is incorrect\n", i);
            errorCount++;
        }

        if (b == 0) {
            if (modulusResults[i] != -1) {
                printf("ERROR: %d Modulus is incorrect\n", i);
                errorCount++;
            }
        }
        else if (a % b != modulusResults[i]) {
            printf("ERROR: %d Modulus is incorrect\n", i);
            errorCount++;
        }

    }

    if (errorCount == 0) {
        printf("Verification Success!\n");
    }

}
