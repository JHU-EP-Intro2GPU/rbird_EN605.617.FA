
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assignment.h"

#include <math.h>
#include <stdio.h>
#include <vector>
#include <stdint.h>

bool disableVerification = false;
bool disablePopulateRandomData = false;

struct StreamTest
{
    // Total number of elements (due to streams, this can be HUGE)
    unsigned int totalElements;
    // Number of elements that stream processes at a time
    unsigned int streamSize;
    // The number of asynchronous streams
    unsigned int numberOfStreams;
    // the block size
    unsigned int blockSize;
};

struct StreamData
{
    HostAndDeviceMemory<float> position, velocity, acceleration, output;

    StreamData(size_t num_elements)
        : position(num_elements), velocity(num_elements), acceleration(num_elements), output(num_elements) {
        // 0 out host memory. If not all streams run, zeroed out memory is correct for validation
        position.clearValues();
        velocity.clearValues();
        acceleration.clearValues();
        output.clearValues();
    }
};

__global__ void calculationPosition(float* finalPosition, const float* initialPosition, const float* velocity, const float* acceleration)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float initPos = initialPosition[tid];
    float veloc = velocity[tid];
    float accel = acceleration[tid];
    //float time = tid; // integer can overflow on the squared value
    int time = tid % 1000; // large values introduce error

    float finalPos = initPos + veloc * time + 0.5 * accel * (time * time);
    finalPosition[tid] = finalPos;
}

void verifyOutput(const StreamData& data, int streamIndex) {
    if (disableVerification)
        return;

    for (int i = 0; i < data.output.size(); i++) {
        float initPos = data.position.host()[i];
        float veloc = data.velocity.host()[i];
        float accel = data.acceleration.host()[i];
        float calculated = data.output.host()[i];
        int time = i % 1000;

        float expectedAnswer = initPos + veloc * time + (0.5) * accel * pow(time, 2);

        if (calculated != expectedAnswer) {
            printf("ERROR (%d, %d): pos: %f vel: %f acel: %f calc: %f, expected: %f\n", streamIndex, i, initPos, veloc, accel, calculated, expectedAnswer);
        }
    }
}

void populateData(StreamData& data) {
    if (disablePopulateRandomData)
        return;

    for (size_t i = 0; i < data.output.size(); i++) {
        data.position.host()[i] = (float(rand() % 20099)) / 10 - 100; //[-100.00, 100.00]
        data.velocity.host()[i] = (float(rand() % 4099)) / 10 - 20; //[-20.00, 20.00]
        data.acceleration.host()[i] = (float(rand() % 1099)) / 10 - 5; //[-5.00, 5.00]
    }
}

void runStreamTests(const StreamTest& test) {
    std::vector<CudaStreamWrapper> streams(test.numberOfStreams);
    //std::vector<HostAndDeviceMemory<float>> dataSets;
    std::vector<StreamData> dataSets;
    std::vector<CudaEventWrapper> events(test.numberOfStreams);

    const size_t blocksPerStream = (test.streamSize + test.blockSize - 1) / test.blockSize;

    // allocate enough space to not require bounds checking in kernel.
    const size_t adjustedNumElementsPerStream = blocksPerStream * test.blockSize;
    for (unsigned int i = 0; i < test.numberOfStreams; i++) {
        // Create memory blocks of size 'streamSize'
        //dataSets.push_back(std::move(HostAndDeviceMemory<float>(adjustedStreamSize)));
        dataSets.emplace_back(adjustedNumElementsPerStream);
        gpuErrchk(cudaEventRecord(events[i].event, streams[i].stream));
    }

    int currentStreamIndex = 0;
    bool streamHasOutputData = false;
    for (size_t elementsProcessed = 0; elementsProcessed < test.totalElements; elementsProcessed += test.streamSize) {
        // make sure stream is ready
        gpuErrchk(cudaEventSynchronize(events[currentStreamIndex].event));
        //gpuErrchk(cudaStreamSynchronize(streams[currentStreamIndex].stream));

        StreamData& data = dataSets[currentStreamIndex];
        cudaStream_t& stream = streams[currentStreamIndex].stream;

        // verify results of previous run
        if (streamHasOutputData) {
            verifyOutput(data, currentStreamIndex);
        }

        // populate new data
        populateData(data);

        // send data
        data.position.transferToDeviceAsync(stream);
        data.velocity.transferToDeviceAsync(stream);
        data.acceleration.transferToDeviceAsync(stream);

        // process data
        calculationPosition<<<blocksPerStream, test.blockSize, 0, stream>>>(
            data.output.device(), data.position.device(), data.velocity.device(), data.acceleration.device());

        // retrieve data back
        data.output.transferToHostAsync(stream);

        gpuErrchk(cudaEventRecord(events[currentStreamIndex].event, stream));
        currentStreamIndex++;
        if (currentStreamIndex == test.numberOfStreams) {
            currentStreamIndex = 0;
            streamHasOutputData = true; // all streams have output data now
        }
    }

    // synchronize the streams
    //for (auto& cudaEvent : events) {
    for (size_t i = 0; i < events.size(); i++) {
        gpuErrchk(cudaEventSynchronize(events[i].event));

        verifyOutput(dataSets[i], i);
    }
}

int main(int argc, char* argv[])
{
    StreamTest testValues;
    testValues.totalElements = 1024;  // Total number of elements (due to streams, this can be HUGE)
    testValues.streamSize = 128;      // Number of elements that stream processes at a time
    testValues.numberOfStreams = 4;   // The number of asynchronous streams
    testValues.blockSize = 32;        // the block size

    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--elements") {
            testValues.totalElements = atoi(argv[++i]);
        }
        else if (arg == "--streamSize") {
            testValues.streamSize = atoi(argv[++i]);
        }
        else if (arg == "--streams") {
            testValues.numberOfStreams = atoi(argv[++i]);
        }
        else if (arg == "--blocksize") {
            testValues.blockSize = atoi(argv[++i]);
        }
        else if (arg == "--disableVerify") {
            disableVerification = true;
        }
        else if (arg == "--disablePopulateData") {
            disablePopulateRandomData = true;
        }
    }

    printf("Elements: %u StreamSize: %u Streams: %u BlockSize: %u\n",
        testValues.totalElements, testValues.streamSize, testValues.numberOfStreams, testValues.blockSize);
    printf("Disable Verify: %d Disable Data Generation: %d\n", (int) disableVerification, (int) disablePopulateRandomData);

    {
        TimeCodeBlockCuda kernelRun("Total processing time");
        runStreamTests(testValues);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    gpuErrchk(cudaDeviceReset());

    return 0;
}

