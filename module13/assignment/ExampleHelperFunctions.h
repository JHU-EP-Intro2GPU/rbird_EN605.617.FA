#pragma once
#ifndef EXAMPLE_HELPER_FUNCTIONS_H
#define EXAMPLE_HELPER_FUNCTIONS_H

#include <chrono>

#define clCheck(ans) openCLAssert((ans), __FILE__, __LINE__);
inline void openCLAssert(cl_int code, const char* file, int line, bool abort = true)
{
    if (code != CL_SUCCESS)
    {
        fprintf(stderr, "openCLAssert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

#define nullCheck(ans) nullAssert((ans), __FILE__, __LINE__);
template<typename T>
inline T* nullAssert(T* ptr, const char* file, int line, bool abort = true)
{
    if (ptr == nullptr)
    {
        fprintf(stderr, "nullAssert: %s %d\n", file, line);
        if (abort) exit(1);
    }

    return ptr;
}

class TimeCodeBlock
{
#ifdef _WIN32
    std::chrono::steady_clock::time_point start;
#else
    std::chrono::system_clock::time_point start;
#endif // _WIN32

    const char* name;
public:
    TimeCodeBlock(const char* blockName) : name(blockName) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~TimeCodeBlock() {
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::microseconds timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
        std::cout << name << " Execution time = " << timeDiff.count() << " microseconds." << std::endl;
    }
};


///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
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
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id* device)
{
    cl_int errNum;
    cl_device_id* devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete[] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* sourceText)
{
    cl_int errNum;
    cl_program program;

    program = clCreateProgramWithSource(context, 1,
        &sourceText,
        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

#endif // !EXAMPLE_HELPER_FUNCTIONS_H

