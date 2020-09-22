#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iostream>


//From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__
void saxpy(int n, float a, float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}


float randFloat()
{
    float a = 5.0;
    return (float(rand()) / float((RAND_MAX)) * a);
}

class TimeCodeBlock
{
    std::chrono::steady_clock::time_point start;
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

bool usePinnedMemory = true;

int main(void)
{
    int N = (1 << 20) * 50;
    float* x, * y, * d_x, * d_y;

    printf("Byte size: %d\n", N * sizeof(float));

    TimeCodeBlock overallTime("Program Execution");

    if (usePinnedMemory)
    {
        TimeCodeBlock pinnedMemoryTime("Allocate pinned memory");
        cudaMallocHost(&x, N * sizeof(float));
        cudaMallocHost(&y, N * sizeof(float));
    }
    else {
        TimeCodeBlock pinnedMemoryTime("Allocate pageable memory");
        x = (float*)malloc(N * sizeof(float));
        y = (float*)malloc(N * sizeof(float));
    }


    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    srand((unsigned int)time(NULL));

    //  for (int i = 0; i < 20; i++) {
    //      printf("%f\n", randFloat());
    //  }
    {
        TimeCodeBlock arrayInitialization("Array Initialization");
        float rand_X = randFloat();
        float rand_Y = randFloat();
        for (int i = 0; i < N; i++) {
            x[i] = rand_X;//randFloat();
            y[i] = rand_Y;//randFloat();
        }
    }

    {
        TimeCodeBlock arrayInitialization("memcpy to device");
        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    }


    // Perform SAXPY on 1M elements
    float constant = randFloat();
    float expectedValue = constant * x[0] + y[0];
    saxpy <<<(N + 255) / 256, 256 >>> (N, constant, d_x, d_y);

    {
        TimeCodeBlock arrayInitialization("memcpy to host");
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    float maxError = 0.0f;
    {
        TimeCodeBlock arrayInitialization("Verification");
        for (int i = 0; i < N; i++) {
            //float expectedValue = constant * x[i] + y[i];
            float error = abs(y[i] - expectedValue);
            maxError = max(maxError, error);
            if (error != 0.0) {
                printf("x[%d]=%f\n", i, y[i]);
                printf("y[%d]=%f\n", i, y[i]);
                printf("y[%d]=%f\n", i, y[i]);
            }
        }
    }

    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    if (usePinnedMemory)
    {
        TimeCodeBlock arrayInitialization("Free pinned memory");
        cudaFreeHost(x);
        cudaFreeHost(y);
    }
    else {
        TimeCodeBlock pinnedMemoryTime("Free pageable memory");
        free(x);
        free(y);
    }
}
