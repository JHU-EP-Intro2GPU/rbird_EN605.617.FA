

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

#include <memory>
#include <cstring>
#include <vector>

struct CommandLineParameters
{
public:
    CommandLineParameters(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--arraysize") == 0) {
                arraySize = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--debug") == 0) {
                debug = true;
            }
        }
    }

    bool debug = false;
    int arraySize = 1000;
};

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

struct OpenCLTestContext
{
public:
    OpenCLTestContext(int argc, char** argv) : params(argc, argv)
    {
        // Create an OpenCL context on first available platform
        context = nullCheck(CreateContext());
    }

    ~OpenCLTestContext() {
        Cleanup(context, memObjects);
    }

    //
    //  Create memory objects used as the arguments to the kernel
    //  The kernel takes three arguments: result (output), a (input),
    //  and b (input)
    //
    bool CreateMemObjects(float* a, float* b, size_t num_elements)
    {
        memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * num_elements, a, NULL);
        memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * num_elements, b, NULL);
        memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(float) * num_elements, NULL, NULL);

        if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
        {
            std::cerr << "Error creating memory objects." << std::endl;
            return false;
        }

        return true;
    }


    void testKernel(const char* kernelName, const char* kernelCode, std::unique_ptr<float[]>& result) {
        std::printf("Testing: %s\n", kernelName);

        if (params.debug)
            std::printf("%s\n", kernelCode);

        // Create command-queue and program on the first device available
        cl_device_id device = 0;
        cl_command_queue commandQueue = nullCheck(CreateCommandQueue(context, &device));
        cl_program program = nullCheck(CreateProgram(context, device, kernelCode));

        // Create OpenCL kernel
        cl_kernel kernel = nullCheck(clCreateKernel(program, kernelName, NULL));


        // Set the kernel arguments (result, a, b)
        clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]));
        clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]));
        clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]));

        size_t globalWorkSize[1] = { params.arraySize };
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
            0, params.arraySize * sizeof(float), result.get(),
            0, NULL, NULL));

        if (params.debug) {
            // Output the result buffer
            for (int i = 0; i < params.arraySize; i++)
            {
                std::cout << result[i] << " ";
            }
            std::cout << std::endl;
        }

        Cleanup(program, kernel, commandQueue);
    };

    cl_context context;
    cl_mem memObjects[3];
    CommandLineParameters params;
};


///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
    OpenCLTestContext testRunner(argc, argv);

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    std::unique_ptr<float[]> result(new float[testRunner.params.arraySize]);
    std::unique_ptr<float[]> a(new float[testRunner.params.arraySize]);
    std::unique_ptr<float[]> b(new float[testRunner.params.arraySize]);

    for (int i = 0; i < testRunner.params.arraySize; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    cl_mem memObjects[3] = { 0, 0, 0 };
    if (!testRunner.CreateMemObjects(a.get(), b.get(), testRunner.params.arraySize))
    {
        return 1;
    }

    std::vector<std::vector<const char*>> testValues = {
        { addKernelFunctionName, addKernelSourceText}, // the first call has some initialization overhead
        { addKernelFunctionName, addKernelSourceText},
        { subKernelFunctionName, subKernelSourceText},
        { multKernelFunctionName, multKernelSourceText},
        { divKernelFunctionName, divKernelSourceText},
        { xorKernelFunctionName, xorKernelSourceText}
    };

    std::printf("Testing %d elements\n", testRunner.params.arraySize);

    for (const auto& test : testValues) {
        testRunner.testKernel(test[0], test[1], result);
    }

    std::cout << "Executed program succesfully." << std::endl;

    return 0;
}
