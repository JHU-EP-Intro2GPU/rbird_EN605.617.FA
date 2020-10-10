#pragma once

#ifndef KERNEL_FUNCTION_DEFINITIONS_H
#define KERNEL_FUNCTION_DEFINITIONS_H

#include "cuda_runtime.h"


// Global Memory Kernels
__global__ void globalMemAdd(int* output, const int* input1, const int* input2, const size_t count);
__global__ void globalMemSub(int* output, const int* input1, const int* input2, const size_t count);
__global__ void globalMemMult(int* output, const int* input1, const int* input2, const size_t count);
__global__ void globalMemMod(int* output, const int* input1, const int* input2, const size_t count);

// Shared Memory Kernels. Requires sizeof(int) * 2 * block_size
__global__ void sharedMemAdd(int* output, const int* input1, const int* input2, const size_t count);
__global__ void sharedMemSub(int* output, const int* input1, const int* input2, const size_t count);
__global__ void sharedMemMult(int* output, const int* input1, const int* input2, const size_t count);
__global__ void sharedMemMod(int* output, const int* input1, const int* input2, const size_t count);

// Register Memory Kernels (uses 1 register)
__global__ void registerMemAdd(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemSub(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMult(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMod(int* output, const int* input1, const int* input2, const size_t count);

// Register memory using 2 registers for loop unrolling
__global__ void registerMemAdd_2(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemSub_2(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMult_2(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMod_2(int* output, const int* input1, const int* input2, const size_t count);

// Register memory using 4 registers for loop unrolling
__global__ void registerMemAdd_4(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemSub_4(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMult_4(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMod_4(int* output, const int* input1, const int* input2, const size_t count);

// Register memory using 8 registers for loop unrolling
__global__ void registerMemAdd_8(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemSub_8(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMult_8(int* output, const int* input1, const int* input2, const size_t count);
__global__ void registerMemMod_8(int* output, const int* input1, const int* input2, const size_t count);

#endif