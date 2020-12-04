#pragma once
#ifndef OPEN_CL_HELPER_H
#define OPEN_CL_HEADER_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <memory>
#include <vector>


static_assert(false, "This is a header that isn't working. It was a fun experiment, but I cannot figure out the nitty gritty details that don't work.");

#define clCheck(ans) { openCLAssert((ans), __FILE__, __LINE__); }
inline void openCLAssert(cl_int code, const char* file, int line, bool abort = true)
{
    if (code != CL_SUCCESS)
    {
        fprintf(stderr, "openCLAssert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

struct OpenCLCommandQueue
{
public:
    OpenCLCommandQueue() {}
    ~OpenCLCommandQueue() {
        if (queue != nullptr)
            clReleaseCommandQueue(queue);
    }

    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
};

struct OpenCLProgram
{
public:
    OpenCLProgram() {}
    ~OpenCLProgram() {
        if (program != nullptr)
            clReleaseProgram(program);
    }

    cl_program program = nullptr;
};

// singleton pattern
class OpenCLContext
{
public:
    static cl_context getContext() {
        if (clContext.context == 0)
            clContext.initContext();

        return clContext.context;
    }

    static void releaseContext() {
        if (clContext.context != 0)
            clReleaseContext(clContext.context);

        clContext.context = 0;
    }

    static OpenCLCommandQueue createCommandQueue(int deviceNum = 0) {
        OpenCLCommandQueue commandQueue;

        // First get the size of the devices buffer
        size_t deviceBufferSize = -1;
        clCheck(clGetContextInfo(getContext(), CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize));

        // Allocate memory for the devices buffer
        std::unique_ptr<cl_device_id[]> devices(new cl_device_id[deviceBufferSize / sizeof(cl_device_id)]);
        clCheck(clGetContextInfo(getContext(), CL_CONTEXT_DEVICES, deviceBufferSize, devices.get(), NULL));

        commandQueue.queue = clCreateCommandQueue(getContext(), devices[deviceNum], 0, NULL);
        if (commandQueue.queue == nullptr)
        {
            std::cerr << "Failed to create commandQueue for device " << deviceNum << std::endl;
            exit(0);
        }

        commandQueue.device = devices[deviceNum];
        return commandQueue;
    }

    static OpenCLProgram createProgramFromFile(cl_device_id device, const char* fileName) {
        std::ifstream kernelFile(fileName, std::ios::in);
        if (!kernelFile.is_open())
        {
            std::cerr << "Failed to open file for reading: " << fileName << std::endl;
            exit(0);
        }

        std::ostringstream oss;
        oss << kernelFile.rdbuf();

        std::string srcStdStr = oss.str();
        return createProgramFromString(device, srcStdStr.c_str());
    }

    static OpenCLProgram createProgramFromString(cl_device_id device, const char* stringVal) {
        OpenCLProgram clProgram;

        clProgram.program = clCreateProgramWithSource(getContext(), 1, &stringVal, NULL, NULL);
        if (clProgram.program == nullptr)
        {
            std::cerr << "Failed to create CL program from source." << std::endl;
            exit(0);
        }

        cl_int errNum = clBuildProgram(clProgram.program, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(clProgram.program, device, CL_PROGRAM_BUILD_LOG,
                sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in kernel: " << std::endl;
            std::cerr << buildLog;
            exit(0);
        }

        return clProgram;
    }


private:
    static OpenCLContext clContext;

    OpenCLContext() {}

    ~OpenCLContext() {
        releaseContext();
    }

    void initContext() {
        cl_uint numPlatforms;
        cl_platform_id firstPlatformId;
        cl_int errNum;

        clCheck(clGetPlatformIDs(1, &firstPlatformId, &numPlatforms));

        cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)firstPlatformId,
            0
        };
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
            NULL, NULL, &errNum);

        if (errNum != CL_SUCCESS)
        {
            std::cout << "Could not create GPU context, trying CPU..." << std::endl;
            context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                NULL, NULL, &errNum);
            clCheck(errNum); // error if neither GPU nor CPU are found
        }
    }

    cl_context context = nullptr;
};

// Declare static storage of context
OpenCLContext OpenCLContext::clContext;


template<typename T>
class OpenCLVector
{
public:

    OpenCLVector(size_t num_elements, cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR) {
        this->num_elements = num_elements;
        hostMemory.reset(new T[num_elements]);

        cl_int error;
        deviceMemory = clCreateBuffer(OpenCLContext::getContext(), flags, sizeof(T) * num_elements, getHostMemoryPtr(), &error);
        clCheck(error);
    }

    void transferMemoryToDevice(OpenCLCommandQueue& queue, bool async=true) {
        auto blocking = (async) ? CL_FALSE : CL_TRUE;
        cl_event event1 = nullptr, event2 = nullptr;
        clCheck(clEnqueueWriteBuffer(queue.queue, deviceMemory, blocking, 0, num_elements * sizeof(T), hostMemory.get(), 0, NULL, NULL));
    }

    void transferMemoryToHost(OpenCLCommandQueue& queue, bool async = true) {
        auto blocking = (async) ? CL_FALSE : CL_TRUE;
        clCheck(clEnqueueReadBuffer(queue.queue, deviceMemory, blocking, 0, num_elements * sizeof(T), hostMemory.get(), 0, NULL, NULL));
    }


    size_t size() const { return num_elements; }
    T& operator[](size_t index) { return hostMemory[index]; }

    cl_mem getDeviceMemory() {
        return deviceMemory;
    }

    cl_mem* getDeviceMemoryPtr() {
        return &deviceMemory;
    }

    T* getHostMemoryPtr() {
        return hostMemory.get();
    }

private:
    std::unique_ptr<T[]> hostMemory;
    size_t num_elements = 0;
    cl_mem deviceMemory = nullptr;
};

#endif // !OPEN_CL_HELPER_H

