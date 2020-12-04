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

const char* xorKernelFunctionName = "xor_kernel";
const char* xorKernelSourceText = R"source(
    __kernel void xor_kernel(__global const float* a,
        __global const float* b,
        __global float* result)
    {
        int gid = get_global_id(0);

        // cast floats to int values to perform XOR (float does not have XOR operator)
        int val1 = a[gid];
        int val2 = b[gid];

        result[gid] = val1 ^ val2;
    }
)source";

#endif // !Kernel_Definitions_H

