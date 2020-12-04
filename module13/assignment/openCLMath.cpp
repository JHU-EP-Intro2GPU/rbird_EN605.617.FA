

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
            else if (strcmp(arg, "--printkernels") == 0) {
                printKernels = true;
            }
            else if (strcmp(arg, "--randomRange") == 0) {
                if (sscanf(argv[++i], "[%d,%d]", &randomMin, &randomMax) != 2) {
                    std::printf("Improper format for --randomRange argument.\n");
                    std::printf("Expected: %s\n", "[%d,%d]");
                    std::printf("Given: %s\n", argv[i]);
                    exit(-1);
                }
            }
            else if (strcmp(arg, "--queues") == 0) {
                numQueues = atoi(argv[++i]);
            }
        }

        if (arraySize % numQueues != 0) {
            std::printf("For simplicity, please use a number of queues that evenly divides the workload/elements\n");
            exit(-1);
        }
    }

    bool useRandomData() const {
        // as long as one value has changed, then use random data
        return randomMin != -1 || randomMax != -1;
    }

    float nextRand() {
        if (randomMin == randomMax) {
            return randomMin;
        }
        else {
            return randomMin + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (randomMax - randomMin)));
        }
    }

    bool debug = false;
    bool printKernels = false;
    int arraySize = 1000;
    int randomMin = -1;
    int randomMax = -1;
    int numQueues = 1;
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

template<typename T>
void printArrayData(const T* buffer, size_t numElements) {
    for (int i = 0; i < numElements; i++)
    {
        std::cout << buffer[i] << " ";
    }
    std::cout << std::endl;
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

    std::vector<cl_mem> createSubbuffers(int offset, int numElements) {
        std::vector<cl_mem> subbuffers;

        cl_buffer_region region;
        region.origin = offset * sizeof(float);
        region.size = numElements * sizeof(float);

        const int numBuffers = 3; // 2 input, 1 output
        for (int i = 0; i < numBuffers; i++) {
            cl_mem_flags subbufferPermissions;
            clCheck(clGetMemObjectInfo(memObjects[i], CL_MEM_FLAGS, sizeof(cl_mem_flags), &subbufferPermissions, nullptr));

            // On my machine, using subbufferPermissions results in a failure, using CL_MEM_READ_ONLY works and seems to allow writes.
            cl_int errNum;
            cl_mem buffer = clCreateSubBuffer(
                memObjects[i],
                CL_MEM_READ_ONLY,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &errNum);
            clCheck(errNum);

            subbuffers.push_back(buffer);
        }

        return subbuffers;
    }


    void testKernel(const char* kernelName, const char* kernelCode, std::unique_ptr<float[]>& result) {
        std::printf("Testing: %s\n", kernelName);

        if (params.printKernels)
            std::printf("%s\n", kernelCode);

        // Create program on the first device available
        cl_device_id device = 0;
        cl_program program = nullCheck(CreateProgram(context, device, kernelCode));

        // Create OpenCL kernel
        cl_kernel kernel = nullCheck(clCreateKernel(program, kernelName, NULL));

        // Create Queues and subbuffers for work
        std::vector<cl_command_queue> queues;

        for (int i = 0; i < params.numQueues; i++) {
            // TODO: create queues and subbuffers
            cl_command_queue commandQueue = nullCheck(CreateCommandQueue(context, &device));
            queues.push_back(commandQueue);
        }

        std::vector<cl_event> events(queues.size(), nullptr);

        size_t kernelWorkload = params.arraySize / queues.size();
        size_t globalWorkSize[1] = { kernelWorkload };
        size_t localWorkSize[1] = { 1 };

        // Queue the kernel up for execution across the array
        {
            TimeCodeBlock kernelRun("Kernel Runtime");
            for (int i = 0; i < queues.size(); i++) {
                cl_command_queue commandQueue = queues[i];
                size_t bufferOffset = i * kernelWorkload;
                std::vector<cl_mem> subbuffers = createSubbuffers(bufferOffset, kernelWorkload);

                // Set the kernel arguments (result, a, b)
                clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &subbuffers[0]));
                clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &subbuffers[1]));
                clCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &subbuffers[2]));

                clCheck(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                    globalWorkSize, localWorkSize,
                    0, NULL, NULL));

                // Read the output buffer back to the Host async
                clCheck(clEnqueueReadBuffer(commandQueue, subbuffers[2], CL_FALSE,
                    0, kernelWorkload * sizeof(float), result.get() + bufferOffset,
                    0, NULL, &events[i]));
            }

            clCheck(clWaitForEvents(events.size(), events.data()));
        }

        if (params.debug) {
            printArrayData(result.get(), params.arraySize);
        }

        Cleanup(program, kernel);
        for (cl_command_queue commandQueue : queues) {
            clCheck(clFinish(commandQueue));
        }
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
    // kernel. First create host memory arrays that will be
    // used to store the arguments to the kernel
    std::unique_ptr<float[]> result(new float[testRunner.params.arraySize]);
    std::unique_ptr<float[]> a(new float[testRunner.params.arraySize]);
    std::unique_ptr<float[]> b(new float[testRunner.params.arraySize]);

    if (testRunner.params.useRandomData()) {
        for (int i = 0; i < testRunner.params.arraySize; i++)
        {
            a[i] = testRunner.params.nextRand();
            b[i] = testRunner.params.nextRand();
        }
    }
    else {
        for (int i = 0; i < testRunner.params.arraySize; i++)
        {
            a[i] = (float)i;
            b[i] = (float)(i * 2);
        }
    }

    if (testRunner.params.debug) {
        std::printf("Input Buffers\na:\n");
        printArrayData(a.get(), testRunner.params.arraySize);
        std::printf("b:\n");
        printArrayData(b.get(), testRunner.params.arraySize);
        std::printf("\n");
    }

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
