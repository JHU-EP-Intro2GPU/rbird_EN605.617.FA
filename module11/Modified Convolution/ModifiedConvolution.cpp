//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "TimeBlock.h"
#include "Signal.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

struct CommandLineArguments {
public:
	CommandLineArguments(int argc, const char* argv[]) {
	}

	int signalRange() {
		return signalMax - signalMin;
	}

	int signalMin = 1;
	int signalMax = 100;

	// the chance that a particular cell will be enabled in the mask
	float maskProbability = 0.4;
	bool debug = false;
};

// 8x8 input, 3x3 mask
Signal<8, 8, 3, 3> smallSignal;
// 49x49 input, 7x7 mask
Signal<49, 49, 7, 7> largeSignal;


cl_float maskGradient[7][7] = {
	{ 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
	{ 0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
	{ 0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
	{ 0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25},
	{ 0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
	{ 0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
	{ 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25}
};

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

template<class T>
void runConvolution(const CommandLineArguments& args, T& signal) {
	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id* platformIDs;
	cl_device_id* deviceIDs;
	cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

	// First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	platformIDs = (cl_platform_id*)alloca(
		sizeof(cl_platform_id) * numPlatforms);

	errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	checkErr(
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
		"clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
			platformIDs[i],
			CL_DEVICE_TYPE_GPU,
			0,
			NULL,
			&numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
		{
			checkErr(errNum, "clGetDeviceIDs");
		}
		else if (numDevices > 0)
		{
			deviceIDs = (cl_device_id*)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices,
				&deviceIDs[0],
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
		}
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

	// Next, create an OpenCL context on the selected platform.  
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platformIDs[i],
		0
	};
	context = clCreateContext(
		contextProperties,
		numDevices,
		deviceIDs,
		&contextCallback,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("ModifiedConvolution.cl");
	checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading ModifiedConvolution.cl");

	std::string srcProg(
		std::istreambuf_iterator<char>(srcFile),
		(std::istreambuf_iterator<char>()));

	const char* src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context,
		1,
		&src,
		&length,
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(
			program,
			deviceIDs[0],
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog),
			buildLog,
			NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
	}

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * signal.inputSignalHeight * signal.inputSignalWidth,
		static_cast<void*>(signal.inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * signal.maskHeight * signal.maskWidth,
		static_cast<void*>(signal.mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * signal.outputSignalHeight * signal.outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &signal.inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &signal.maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { signal.outputSignalWidth, signal.outputSignalHeight };
	const size_t localWorkSize[2] = { 1, 1 };

	// Queue the kernel up for execution across the array
	errNum = clEnqueueNDRangeKernel(
		queue,
		kernel,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");

	errNum = clEnqueueReadBuffer(
		queue,
		outputSignalBuffer,
		CL_TRUE,
		0,
		sizeof(cl_uint) * signal.outputSignalHeight * signal.outputSignalHeight,
		signal.outputSignal,
		0,
		NULL,
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
}

template<class T>
void testConvolution(const CommandLineArguments& args, T& signal)
{
	{
		char blockName[1000];
		sprintf(blockName, "Total Run %d x %d", signal.inputSignalWidth, signal.inputSignalHeight);
		TimeCodeBlock totalProcess(blockName);
		runConvolution(args, signal);
	}

	int maskMidRow = signal.maskHeight / 2;
	int maskMidCol = signal.maskWidth / 2;

	std::printf("Distances:\n");
	for (int r = 0; r < signal.maskHeight; r++) {
		int rowDistance = pow(r - maskMidRow, 2);
		for (int c = 0; c < signal.maskWidth; c++) {
			if (c != 0)
				std::printf(" ");
			int colDistance = pow(c - maskMidCol, 2);
			int distance = sqrt(rowDistance + colDistance) + 1;
			std::printf("%d", distance);
		}

		std::printf("\n");
	}

	std::printf("\n");

	if (args.debug) {
		// output the expected computation of output[0][0]
		int maskSum = 0;
		for (int r = 0; r < signal.maskHeight; r++) {
			int rowSum = 0;
			for (int c = 0; c < signal.maskWidth; c++) {
				if (c != 0)
					std::printf("+");
				std::printf("%d ", signal.inputSignal[r][c]);
				rowSum += signal.inputSignal[r][c];
			}

			maskSum += rowSum;
			std::printf("= %d,\t%d\n", rowSum, maskSum);
		}

		// Output the result buffer
		for (int y = 0; y < signal.outputSignalHeight; y++)
		{
			for (int x = 0; x < signal.outputSignalWidth; x++)
			{
				std::cout << signal.outputSignal[y][x] << " ";
			}
			std::cout << std::endl;
		}
	}

}

void populateDefaultSmallSignal() {
	std::vector<std::vector<cl_int>> signalValues = {
		{3, 1, 1, 4, 8, 2, 1, 3},
		{4, 2, 1, 1, 2, 1, 2, 3},
		{4, 4, 4, 4, 3, 2, 2, 2},
		{9, 8, 3, 8, 9, 0, 0, 0},
		{9, 3, 3, 9, 0, 0, 0, 0},
		{0, 9, 0, 8, 0, 0, 0, 0},
		{3, 0, 8, 8, 9, 4, 4, 4},
		{5, 9, 8, 1, 8, 1, 1, 1}
	};

	for (int r = 0; r < signalValues.size(); r++) {
		for (int c = 0; c < signalValues[r].size(); c++) {
			smallSignal.inputSignal[r][c] = signalValues[r][c];
		}
	}

	std::vector <std::vector<cl_int>> maskValues = {
		{1, 1, 1}, {1, 0, 1}, {1, 1, 1},
	};

	for (int r = 0; r < maskValues.size(); r++) {
		for (int c = 0; c < maskValues[r].size(); c++) {
			smallSignal.mask[r][c] = maskValues[r][c];
		}
	}
}

///
//	main() for Convoloution example
//
int main(int argc, const char* argv[])
{
	CommandLineArguments args(argc, argv);

	populateDefaultSmallSignal();
	largeSignal.populateData();

	std::printf("The first kernel run contains overhead to initialize OpenCL. Ignore this first output.\n");
	testConvolution(args, smallSignal);
	testConvolution(args, smallSignal);
	testConvolution(args, largeSignal);

    std::cout << std::endl << "Executed program succesfully." << std::endl;

	return 0;
}
