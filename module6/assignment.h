#pragma once

#ifndef ASSIGNMENT_H
#define ASSIGNMENT_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <chrono>


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

class TimeCodeBlock
{
#ifdef _WIN32
    std::chrono::steady_clock::time_point start;
#else
    std::chrono::system_clock::time_point start;
#endif // _WIN32

    const char* name;
public:
    TimeCodeBlock(const char* blockName) : name(blockName) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~TimeCodeBlock() {
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::microseconds timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
        std::cout << name << " Execution time = " << timeDiff.count() << " microseconds." << std::endl;
    }
};

#endif

