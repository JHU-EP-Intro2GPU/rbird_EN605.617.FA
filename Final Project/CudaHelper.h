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

class TimeCodeBlockCuda
{
    // events for timing
    cudaEvent_t startEvent, stopEvent;

    const char* name;
public:
    TimeCodeBlockCuda(const char* blockName) : name(blockName) {
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
            printf("%s Execution time = %d seconds %d milliseconds %d microseconds\n", name, timeSeconds, time_millis, time_micros);
        }
        else if (time_millis > 0) {
            printf("%s Execution time = %d milliseconds %d microseconds\n", name, time_millis, time_micros);
        }
        else {
            printf("%s Execution time = %d microseconds\n", name, time_micros);
        }

        gpuErrchk(cudaEventDestroy(startEvent));
        gpuErrchk(cudaEventDestroy(stopEvent));
    }
};


template <typename T>
class HostAndDeviceMemory
{
    T* host_ptr;
    T* device_ptr;
    size_t _size;

public:
    HostAndDeviceMemory() : host_ptr(nullptr), device_ptr(nullptr), _size(-1) {
    }

    HostAndDeviceMemory(HostAndDeviceMemory&& memory) {
        host_ptr = std::move(memory.host_ptr);
        device_ptr = std::move(memory.device_ptr);
        _size = std::move(memory._size);

        memory.host_ptr = nullptr;
        memory.device_ptr = nullptr;
        memory._size = -1;
    }

    HostAndDeviceMemory(size_t bufferCount) : _size(bufferCount) {
        allocate(bufferCount);
    }

    void allocate(size_t bufferCount) {
        _size = bufferCount;
        size_t bytes = _size * sizeof(T);
        gpuErrchk(cudaMallocHost(&host_ptr, bytes));
        gpuErrchk(cudaMalloc((void**)&device_ptr, bytes));
    }

    void transferToDevice() {
        gpuErrchk(cudaMemcpy(device_ptr, host_ptr, _size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void transferToHost() {
        gpuErrchk(cudaMemcpy(host_ptr, device_ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
    }


    void transferToDeviceAsync(cudaStream_t& stream) {
        gpuErrchk(cudaMemcpyAsync(device_ptr, host_ptr, _size * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void transferToHostAsync(cudaStream_t& stream) {
        gpuErrchk(cudaMemcpyAsync(host_ptr, device_ptr, _size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    void clearValues() {
        for (size_t i = 0; i < _size; i++) {
            host_ptr[i] = 0;
        }
    }

    void deallocate() {
        gpuErrchk(cudaFreeHost(host_ptr));
        gpuErrchk(cudaFree(device_ptr));

        host_ptr = nullptr;
        device_ptr = nullptr;
        _size = -1;
    }

    virtual ~HostAndDeviceMemory() {
        deallocate();
    }

    inline T* host() const { return host_ptr; }
    inline T* device() const { return device_ptr; }
    inline size_t size() const { return _size; }
};

// A helper wrapper to auto clean up cuda stream objects
struct CudaStreamWrapper
{
    cudaStream_t stream;
public:
    CudaStreamWrapper() {
        gpuErrchk(cudaStreamCreate(&stream));
    }

    ~CudaStreamWrapper() {
        gpuErrchk(cudaStreamDestroy(stream));
    }
};

struct CudaEventWrapper
{
    cudaEvent_t event;
public:
    CudaEventWrapper() {
        gpuErrchk(cudaEventCreate(&event));
    }

    ~CudaEventWrapper() {
        gpuErrchk(cudaEventDestroy(event));
    }
};



#endif

