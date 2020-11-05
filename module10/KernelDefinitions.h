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

#endif // !Kernel_Definitions_H

