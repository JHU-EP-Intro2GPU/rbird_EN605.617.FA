
#include <iostream>
#include <fstream>
#include <sstream>

#include "OpenCLHelper.h"

///
//  Constants
//
const int ARRAY_SIZE = 1000;

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


///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
    cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }

    if (kernel != 0)
        clReleaseKernel(kernel);
}

///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_program program = 0;
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = OpenCLContext::getContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    OpenCLCommandQueue queue = OpenCLContext::createCommandQueue();

    // Create OpenCL program from HelloWorld.cl kernel source
    OpenCLProgram clProgram = OpenCLContext::createProgramFromString(queue.device, addKernelSourceText);

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    OpenCLVector<float> a(ARRAY_SIZE), b(ARRAY_SIZE), result(ARRAY_SIZE, CL_MEM_READ_WRITE);

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    a.transferMemoryToDevice(queue);
    b.transferMemoryToDevice(queue);

    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(clProgram.program, addKernelFunctionName, &errNum);

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), a.getDeviceMemoryPtr());
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), b.getDeviceMemoryPtr());
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), result.getDeviceMemoryPtr());

    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(queue.queue, kernel, 1, NULL,
        globalWorkSize, localWorkSize,
        0, NULL, NULL);

    // Read the output buffer back to the Host
    result.transferMemoryToHost(queue, false /* async */);

    // Output the result buffer
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;


    return 0;
}
