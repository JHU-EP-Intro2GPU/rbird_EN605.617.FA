
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assignment.h"
#include "kernels.h"

#include <cstdlib>
#include <stdio.h>
#include <string.h>

struct CommandLineArgs {
public:
    CommandLineArgs(int argc, const char* argv[]) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--elements") == 0) {
                elements = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--blocksize") == 0) {
                blocksize = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--random") == 0) {
                randomSeed = true;
            }
            else if (strcmp(arg, "--debug") == 0) {
                debug = true;
            }
        }

        if (!IsPowerOfTwo(blocksize)) {
            printf("Please enter a blocksize that is a power of 2 (this is for simplicity of reduction code).\n");
            exit(0);
        }
    }

    int totalBlocks() const {
        return (elements + blocksize - 1) / blocksize;
    }

    bool IsPowerOfTwo(int x)
    {
        return (x != 0) && ((x & (x - 1)) == 0);
    }

    int elements = 32;
    int blocksize = 8;
    bool randomSeed = false;
    bool debug = false;
};

void testMonteCarloAlgorithm(const CommandLineArgs& testArgs);

int main(int argc, const char* argv[])
{
    CommandLineArgs testArgs(argc, argv);

    testMonteCarloAlgorithm(testArgs);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    gpuErrchk(cudaDeviceReset());

    return 0;
}

void testMonteCarloAlgorithm(const CommandLineArgs& testArgs)
{
    const int numBlocks = testArgs.totalBlocks();
    // allocates more buffer size if needed so we don't have to do bounds checking
    const size_t allocateElements = numBlocks * testArgs.blocksize;
    const int sharedMemorySize = testArgs.blocksize * sizeof(int);

    HostAndDeviceMemory<float> X(allocateElements), Y(allocateElements);
    HostAndDeviceMemory<unsigned int> totalCount(1);

    totalCount.host()[0] = 0;
    totalCount.transferToDevice();

    printf("Monte Carlo Algorithm\n");
    printf("elements: %d blocksize: %d\n", testArgs.elements, testArgs.blocksize);

    unsigned long long seed = 1;

    if (testArgs.randomSeed)
        seed = time(NULL);

    {
        TimeCodeBlockCuda generateRandom("Monte Carlo Algorithm");

        generatePoints<<<numBlocks, testArgs.blocksize>>>(X.device(), Y.device(), seed);
        gpuErrchk(cudaGetLastError());

        countPointsWithinCircle<<<numBlocks, testArgs.blocksize, sharedMemorySize>>>(X.device(), Y.device(), totalCount.device());
        gpuErrchk(cudaGetLastError());
    }

    if (testArgs.debug) {
        X.transferToHost();
        Y.transferToHost();

        int expectedTotalCount = 0;

        for (int i = 0; i < testArgs.elements; i++) {
            float x = X.host()[i];
            float y = Y.host()[i];

            if ((pow(x, 2) + pow(y, 2)) <= 1.0)
                expectedTotalCount++;

            if (i != 0 && i % testArgs.blocksize == 0)
                printf("\n");

            printf("(%f, %f)\t", x, y);
        }
        printf("\n");

        printf("Expected points in circle: %d\n", expectedTotalCount);
    }

    totalCount.transferToHost();
    const float inCircle = totalCount.host()[0];

    float piEstimate = 4 * inCircle / testArgs.elements;

    printf("Total points within circle: %d\n", totalCount.host()[0]);
    printf("pi estimate: %f\n", piEstimate);

}

