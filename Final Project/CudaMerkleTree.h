#pragma once
#ifndef CUDA_MERKLE_TREE_H
#define CUDA_MERKLE_TREE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

struct SHA256Digest {
    // 32 bits * 8 variables = 256 bits total
    uint32_t h0, h1, h2, h3, h4, h5, h6, h7;
};

#endif // !CUDA_MERKLE_TREE_H

