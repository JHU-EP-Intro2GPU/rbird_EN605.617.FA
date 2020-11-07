#pragma once
#ifndef DATA_BLOCKS_H
#define DATA_BLOCKS_H

#include "cuda_runtime.h"

#include <cstdint>

__device__
struct DataBlock_512_bit {
    // 32 bit integers * 16 integers = 512 bits
    uint32_t h0, h1, h2, h3, h4, h5, h6, h7;
    uint32_t h8, h9, h10, h11, h12, h13, h14, h15;
 
    __device__ void convertToPaddedBlock(uint64_t messageLength) {
        // append a single '1' bit
        // append K '0' bits, where K is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
        // append L as a 64 - bit big - endian integer, making the total post - processed length a multiple of 512 bits
        h0 = 1 << 31;
        h1 = 0;
        h2 = 0;
        h3 = 0;
        h4 = 0;
        h5 = 0;
        h6 = 0;
        h7 = 0;
        h8 = 0;
        h9 = 0;
        h10 = 0;
        h11 = 0;
        h12 = 0;
        h13 = 0;

        h14 = messageLength >> 32; // the high 32 bits
        h15 = messageLength; // the low 32 bits
    }
};


__device__ uint32_t rotateBitsRight(uint32_t val, int rotateBy) {
    return (val >> rotateBy) | (val << (32 - rotateBy));
}

__device__
struct DataBlock_2048_bit {
    // 32 bit integers * 64 integers = 2048 bits
    //uint32_t h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15;
    //uint32_t h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31;
    //uint32_t h32, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42, h43, h44, h45, h46, h47;
    //uint32_t h48, h49, h50, h51, h52, h53, h54, h55, h56, h57, h58, h59, h60, h61, h62, h63;

    // TODO: Can we get data to live in shared memory?
    uint32_t data[64];

    __device__ void processData(const DataBlock_512_bit& src) {
        // this could be very slow since this may no longer be register memory
        // consider making this parallelizeable among different threads
        data[0] = src.h0;
        data[1] = src.h1;
        data[2] = src.h2;
        data[3] = src.h3;

        data[4] = src.h4;
        data[5] = src.h5;
        data[6] = src.h6;
        data[7] = src.h7;

        data[8] = src.h8;
        data[9] = src.h9;
        data[10] = src.h10;
        data[11] = src.h11;

        data[12] = src.h12;
        data[13] = src.h13;
        data[14] = src.h14;
        data[15] = src.h15;

        // consider parallelizing this among different threads (may not be possible, as there are tight data dependencies
        for (int i = 16; i < 64; i++) {
            uint32_t prevVal0 = data[i - 15];
            uint32_t prevVal1 = data[i - 2];
            uint32_t s0 = rotateBitsRight(prevVal0, 7) ^ rotateBitsRight(prevVal0, 18) ^ (prevVal0 >> 3);
            uint32_t s1 = rotateBitsRight(prevVal1, 17) ^ rotateBitsRight(prevVal1, 19) ^ (prevVal1 >> 10);

            data[i] = data[i - 16] + s0 + data[i - 7] + s1;
        }
    }
};


#endif // !DATA_BLOCKS_H

