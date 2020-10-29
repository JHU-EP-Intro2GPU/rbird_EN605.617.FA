#pragma once

#ifndef ASSIGNMENT_H
#define ASSIGNMENT_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <string>


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

class TimeCodeBlockCuda
{
    // events for timing
    cudaEvent_t startEvent, stopEvent;

    std::string name;
public:
    TimeCodeBlockCuda(std::string blockName) : name(std::move(blockName)) {
        gpuErrchk(cudaEventCreate(&startEvent));
        gpuErrchk(cudaEventCreate(&stopEvent));

        gpuErrchk(cudaEventRecord(startEvent, 0));
    }

    ~TimeCodeBlockCuda() {
        gpuErrchk(cudaEventRecord(stopEvent, 0));
        gpuErrchk(cudaEventSynchronize(stopEvent));

        float time_ms;
        gpuErrchk(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));

        unsigned long total_micros = (unsigned long)(time_ms * 1000);
        int time_micros = total_micros % 1000;

        unsigned long total_millis = total_micros / 1000;
        int time_millis = total_millis % 1000;

        int timeSeconds = total_millis / 1000;

        if (timeSeconds > 0) {
            printf("%s Execution time = %d seconds %d milliseconds %d microseconds\n", name.c_str(), timeSeconds, time_millis, time_micros);
        }
        else if (time_millis > 0) {
            printf("%s Execution time = %d milliseconds %d microseconds\n", name.c_str(), time_millis, time_micros);
        }
        else {
            printf("%s Execution time = %d microseconds\n", name.c_str(), time_micros);
        }

        gpuErrchk(cudaEventDestroy(startEvent));
        gpuErrchk(cudaEventDestroy(stopEvent));
    }
};


#endif

