

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "KernelDefinitions.h"
#include "ExampleHelperFunctions.h"

#include <vector>

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context) {
    if (context != 0)
        clReleaseContext(context);
}

void Cleanup(cl_command_queue commandQueue) {
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
}

void Cleanup(cl_program program) {
    if (program != 0)
        clReleaseProgram(program);
}

void Cleanup(cl_kernel kernel) {
    if (kernel != 0)
        clReleaseKernel(kernel);
}

void Cleanup(cl_mem memObjects[3]) {
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
}

// flexible Cleanup definition
template <typename T, typename... Args>
void Cleanup(T item, Args... args)
{
    Cleanup(item);
    Cleanup(args...);
}


bool debug = false;

///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    // Create an OpenCL context on first available platform
    cl_context context = nullCheck(CreateContext());

    // Create a command-queue on the first device available
    // on the created context
    cl_device_id device = 0;
    cl_command_queue commandQueue = nullCheck(CreateCommandQueue(context, &device));

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    cl_mem memObjects[3] = { 0, 0, 0 };
    if (!CreateMemObjects(context, memObjects, a, b, ARRAY_SIZE))
    {
        Cleanup(context, commandQueue, memObjects);
        return 1;
    }

    std::vector<std::vector<const char*>> testValues = {
        { addKernelFunctionName, addKernelSourceText}, // the first call has some initialization overhead
        { addKernelFunctionName, addKernelSourceText},
        { subKernelFunctionName, subKernelSourceText},
        { multKernelFunctionName, multKernelSourceText},
        { divKernelFunctionName, divKernelSourceText},
        { powKernelFunctionName, powKernelSourceText}
    };

    // test lambda captures local variables (context for example)
    auto testKernel = [&](const char* kernelName, const char* kernelCode) {
        std::printf("Testing: %s\n", kernelName);

        if (debug)
            std::printf("%s\n", kernelCode);

        // Create OpenCL program from HelloWorld.cl kernel source
        cl_program program = nullCheck(CreateProgram(context, device, kernelCode));

        // Create OpenCL kernel
        cl_kernel kernel = nullCheck(clCreateKernel(program, kernelName, NULL));

        // Set the kernel arguments (result, a, b)
        clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]));
        clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]));
        clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]));

        size_t globalWorkSize[1] = { ARRAY_SIZE };
        size_t localWorkSize[1] = { 1 };

        // Queue the kernel up for execution across the array
        {
            TimeCodeBlock kernelRun("Kernel Runtime");
            clCheck(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                globalWorkSize, localWorkSize,
                0, NULL, NULL));
            clCheck(clFinish(commandQueue));
        }

        // Read the output buffer back to the Host
        clCheck(clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
            0, ARRAY_SIZE * sizeof(float), result,
            0, NULL, NULL));

        if (debug) {
            // Output the result buffer
            for (int i = 0; i < ARRAY_SIZE; i++)
            {
                std::cout << result[i] << " ";
            }
            std::cout << std::endl;
        }

        Cleanup(program, kernel);
    };

    for (const auto& test : testValues) {
        testKernel(test[0], test[1]);
    }

    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, memObjects);

    return 0;
}
