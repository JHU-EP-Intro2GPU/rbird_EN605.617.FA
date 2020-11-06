#pragma once
#ifndef Kernel_Definitions_H
#define Kernel_Definitions_H

const char* addKernelFunctionName = "add_kernel";
const char* addKernelSourceText = R"source(
    __kernel void add_kernel(__global const float* a,
        __global const float* b,
        __global float* result)
    {
        int gid = get_global_id(0);

        result[gid] = a[gid] + b[gid];
    }
)source";

const char* subKernelFunctionName = "sub_kernel";
const char* subKernelSourceText = R"source(
    __kernel void sub_kernel(__global const float* a,
        __global const float* b,
        __global float* result)
    {
        int gid = get_global_id(0);

        result[gid] = a[gid] - b[gid];
    }
)source";

const char* multKernelFunctionName = "mult_kernel";
const char* multKernelSourceText = R"source(
    __kernel void mult_kernel(__global const float* a,
        __global const float* b,
        __global float* result)
    {
        int gid = get_global_id(0);

        result[gid] = a[gid] * b[gid];
    }
)source";

const char* divKernelFunctionName = "div_kernel";
const char* divKernelSourceText = R"source(
    __kernel void div_kernel(__global const float* a,
        __global const float* b,
        __global float* result)
    {
        int gid = get_global_id(0);

        result[gid] = a[gid] / b[gid];
    }
)source";

const char* powKernelFunctionName = "pow_kernel";
const char* powKernelSourceText = R"source(
    __kernel void pow_kernel(__global const float* a,
        __global const float* b,
        __global float* result)
    {
        int gid = get_global_id(0);

        int value = 1;
        int base = a[gid];
        int power = b[gid];
        for (int count = 0; count < power; count++)
            value *= base;

        result[gid] = value;
    }
)source";

#endif // !Kernel_Definitions_H

