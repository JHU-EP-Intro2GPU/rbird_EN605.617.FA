#pragma once

#ifndef KERNELS_H
#define KERNELS_H

__global__ void generatePoints(float* X, float* Y, unsigned long long seed);
__global__ void countPointsWithinCircle(const float* X, const float* Y, unsigned int* output);



#endif

