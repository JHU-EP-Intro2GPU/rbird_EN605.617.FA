


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "assignment.h"
#include "KernelFunctionDefinitions.h"

#include <cstring>
#include <cstdlib>

#include <stdio.h>

int arraySize = 512;
int blockSize = 32;


enum TestKernelType {
    GLOBAL_MEM_ADD, GLOBAL_MEM_SUB, GLOBAL_MEM_MULT, GLOBAL_MEM_MOD,
    SHARED_MEM_ADD, SHARED_MEM_SUB, SHARED_MEM_MULT, SHARED_MEM_MOD,
    REGISTER_MEM_ADD, REGISTER_MEM_SUB, REGISTER_MEM_MULT, REGISTER_MEM_MOD,
    REGISTER_MEM_2_ADD, REGISTER_MEM_2_SUB, REGISTER_MEM_2_MULT, REGISTER_MEM_2_MOD,
};

void populateTestData() {
}

void validateCorrectness() {
}

void resetOutputBufferData() {
    // clear output buffers
//    gpuErrchk(cudaMemset(d_output, 0, sizeof(gmem_output)));
//    memset(host_output, 0, sizeof(host_output));
}

void testKernelRun(TestKernelType kernelType, const char* description) {

    {
        TimeCodeBlock kernelRunMeasurement(description);

        switch (kernelType)
        {
        default:
            break;
        }

        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    validateCorrectness();
    resetOutputBufferData();
}

void testKernels() {
//    printf("Arraysize: %d Blocksize: %d Iterations: %d\n", arraySize, blockSize, ITERATIONS);
    populateTestData();

    resetOutputBufferData();

    /*
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
    */
}


int main(int argc, char* argv[])
{
    testKernels();

    return 0;
}

