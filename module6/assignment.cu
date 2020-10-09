


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "assignment.h"
#include "KernelFunctionDefinitions.h"

#include <cstring>
#include <cstdlib>

#include <stdio.h>

int arraySize = 512;
int blockSize = 32;

DeviceMemory<int> d_source1, d_source2, d_output;

std::unique_ptr<int[]> host_source1, host_source2, host_output;


enum TestKernelType {
    GLOBAL_MEM_ADD, GLOBAL_MEM_SUB, GLOBAL_MEM_MULT, GLOBAL_MEM_MOD,
    SHARED_MEM_ADD, SHARED_MEM_SUB, SHARED_MEM_MULT, SHARED_MEM_MOD,
    REGISTER_MEM_ADD, REGISTER_MEM_SUB, REGISTER_MEM_MULT, REGISTER_MEM_MOD,
    REGISTER_MEM_2_ADD, REGISTER_MEM_2_SUB, REGISTER_MEM_2_MULT, REGISTER_MEM_2_MOD,
};

enum MathOperation {
    ADD, SUB, MULT, MOD
};

char OpToChar(MathOperation operation) {
    switch (operation) {
    case MathOperation::ADD:
        return '+';
    case MathOperation::SUB:
        return '-';
    case MathOperation::MULT:
        return '*';
    case MathOperation::MOD:
        return '%';
    }
}

void populateTestData() {
    d_output.allocate(arraySize);
    d_source1.allocate(arraySize);
    d_source2.allocate(arraySize);

    host_output.reset(new int[arraySize]);
    host_source1.reset(new int[arraySize]);
    host_source2.reset(new int[arraySize]);

    for (int i = 0; i < arraySize; i++) {
        host_source1[i] = rand() % 1000;
        host_source2[i] = rand() % 1000;
    }

    cudaMemcpy(d_source1.ptr(), host_source1.get(), arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source2.ptr(), host_source2.get(), arraySize * sizeof(int), cudaMemcpyHostToDevice);
}

void validateCorrectness(MathOperation operation) {
    cudaMemcpy(host_output.get(), d_output.ptr(), arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; i++) {
        int expectedAnswer;
        switch (operation) {
        case MathOperation::ADD:
            expectedAnswer = host_source1[i] + host_source2[i];
            break;
        case MathOperation::SUB:
            expectedAnswer = host_source1[i] - host_source2[i];
            break;
        case MathOperation::MULT:
            expectedAnswer = host_source1[i] * host_source2[i];
            break;
        case MathOperation::MOD:
            expectedAnswer = host_source2[i] == 0? -1 : host_source1[i] % host_source2[i];
            break;
        }

        // DEBUG
//        printf("%d: %4d %c %4d = %4d, got: %4d\n", i, host_source1[i], OpToChar(operation), host_source2[i], expectedAnswer, host_output[i]);

        if (host_output[i] != expectedAnswer) {
            printf("%d: ERROR! %4d %c %4d = %4d, got: %4d\n", i, host_source1[i], OpToChar(operation), host_source2[i], expectedAnswer, host_output[i]);
        }
    }
}

void resetOutputBufferData() {
    // clear output buffers
    gpuErrchk(cudaMemset(d_output.ptr(), 0, arraySize * sizeof(int)));
    memset(host_output.get(), 0, arraySize * sizeof(int));
}

void testKernelRun(TestKernelType kernelType, const char* description, MathOperation operation) {
    int numBlocks = (arraySize + blockSize - 1) / blockSize;

    // each thread in a block stores two values in shared memory
    int sharedMemoryBytes = blockSize * sizeof(int) * 2;

    {
        TimeCodeBlock kernelRunMeasurement(description);

        switch (kernelType)
        {
        // Global memory tests
        case GLOBAL_MEM_ADD:
            globalMemAdd<<< numBlocks, blockSize>>>(d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case GLOBAL_MEM_SUB:
            globalMemSub <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case GLOBAL_MEM_MULT:
            globalMemMult <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case GLOBAL_MEM_MOD:
            globalMemMod <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;

        // Shared memory tests
        case SHARED_MEM_ADD:
            sharedMemAdd <<< numBlocks, blockSize, sharedMemoryBytes >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case SHARED_MEM_SUB:
            sharedMemSub <<< numBlocks, blockSize, sharedMemoryBytes >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case SHARED_MEM_MULT:
            sharedMemMult <<< numBlocks, blockSize, sharedMemoryBytes >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case SHARED_MEM_MOD:
            sharedMemMod <<< numBlocks, blockSize, sharedMemoryBytes >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;

        // 1 Register memory tests
        case REGISTER_MEM_ADD:
            registerMemAdd <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case REGISTER_MEM_SUB:
            registerMemSub <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case REGISTER_MEM_MULT:
            registerMemMult <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case REGISTER_MEM_MOD:
            registerMemMod <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;

        // 2 Register memory tests
        case REGISTER_MEM_2_ADD:
            numBlocks = ((arraySize / 2) + blockSize - 1) / blockSize;
            registerMemAdd_2 <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case REGISTER_MEM_2_SUB:
            numBlocks = ((arraySize / 2) + blockSize - 1) / blockSize;
            registerMemSub_2 <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case REGISTER_MEM_2_MULT:
            numBlocks = ((arraySize / 2) + blockSize - 1) / blockSize;
            registerMemMult_2 <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;
        case REGISTER_MEM_2_MOD:
            numBlocks = ((arraySize / 2) + blockSize - 1) / blockSize;
            registerMemMod_2 <<< numBlocks, blockSize >>> (d_output.ptr(), d_source1.ptr(), d_source2.ptr(), arraySize);
            break;

        default:
            break;
        }

        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    validateCorrectness(operation);
    resetOutputBufferData();
}

void testKernels() {
//    printf("Arraysize: %d Blocksize: %d Iterations: %d\n", arraySize, blockSize, ITERATIONS);
    populateTestData();

    resetOutputBufferData();

    printf("--------------- GLOBAL MEMORY TESTS -------------------------\n");

    testKernelRun(TestKernelType::GLOBAL_MEM_ADD, "Global Memory Add Kernel", MathOperation::ADD);
    testKernelRun(TestKernelType::GLOBAL_MEM_SUB, "Global Memory Sub Kernel", MathOperation::SUB);
    testKernelRun(TestKernelType::GLOBAL_MEM_MULT, "Global Memory Mult Kernel", MathOperation::MULT);
    testKernelRun(TestKernelType::GLOBAL_MEM_MOD, "Global Memory Mod Kernel", MathOperation::MOD);


    printf("\n--------------- SHARED MEMORY TESTS -------------------------\n");
    testKernelRun(TestKernelType::SHARED_MEM_ADD, "Shared Memory Add Kernel", MathOperation::ADD);
    testKernelRun(TestKernelType::SHARED_MEM_SUB, "Shared Memory Sub Kernel", MathOperation::SUB);
    testKernelRun(TestKernelType::SHARED_MEM_MULT, "Shared Memory Mult Kernel", MathOperation::MULT);
    testKernelRun(TestKernelType::SHARED_MEM_MOD, "Shared Memory Mod Kernel", MathOperation::MOD);


    printf("\n--------------- REGISTER (1) MEMORY TESTS -------------------------\n");
    testKernelRun(TestKernelType::REGISTER_MEM_ADD, "Register Memory Add Kernel", MathOperation::ADD);
    testKernelRun(TestKernelType::REGISTER_MEM_SUB, "Register Memory Sub Kernel", MathOperation::SUB);
    testKernelRun(TestKernelType::REGISTER_MEM_MULT, "Register Memory Mult Kernel", MathOperation::MULT);
    testKernelRun(TestKernelType::REGISTER_MEM_MOD, "Register Memory Mod Kernel", MathOperation::MOD);

    printf("\n--------------- REGISTER (2) MEMORY TESTS -------------------------\n");
    testKernelRun(TestKernelType::REGISTER_MEM_2_ADD, "Register (2) Memory Add Kernel", MathOperation::ADD);
    testKernelRun(TestKernelType::REGISTER_MEM_2_SUB, "Register (2) Memory Sub Kernel", MathOperation::SUB);
    testKernelRun(TestKernelType::REGISTER_MEM_2_MULT, "Register (2) Memory Mult Kernel", MathOperation::MULT);
    testKernelRun(TestKernelType::REGISTER_MEM_2_MOD, "Register (2) Memory Mod Kernel", MathOperation::MOD);

}


int main(int argc, char* argv[])
{
    testKernels();

    return 0;
}

