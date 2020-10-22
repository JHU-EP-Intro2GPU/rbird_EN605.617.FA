
#include "kernels.h"

#include <curand.h>
#include <curand_kernel.h>

// cudRand documentation has example of Monte Carlo algorithm:
// https://docs.nvidia.com/cuda/curand/device-api-overview.html
//
// However, I am doing my own implementation using the following link as my guide:
// https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/



__global__ void generatePoints(float* X, float* Y, unsigned long long seed)
{
    curandState_t state;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    curand_init(seed, tid, 0, &state);

    // curand returns value between 0.0 and 1.0
    // adjust to generate points within [-1.0, 1.0]
    float x = curand_uniform(&state) * 2 - 1;
    float y = curand_uniform(&state) * 2 - 1;

    X[tid] = x;
    Y[tid] = y;
}

// output is a single variable that requires atomic addition
__global__ void countPointsWithinCircle(const float* X, const float* Y, unsigned int* output)
{
    __shared__ extern int counts[];
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const float x = X[tid];
    const float y = Y[tid];

    // no sqrt?
    float distanceFromCenter = (x * x) + (y * y);

    // cheat for counting if distance < 1
    int count = 1 - ((int)distanceFromCenter);

    counts[threadIdx.x] = count;

    __syncthreads();

    // perform block reduction/summation
    int offset = blockDim.x / 2;

    for (; offset > 0; offset /= 2) {
        if (threadIdx.x < offset && (threadIdx.x + offset) < blockDim.x) {
            counts[threadIdx.x] += counts[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, counts[0]);
    }
}

