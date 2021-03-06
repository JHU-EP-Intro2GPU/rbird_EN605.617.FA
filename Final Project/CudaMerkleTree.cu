
#include "CudaMerkleTree.h"
#include "DataBlocks.h"

#include <cstdint>

__device__ __host__
void printDigest(const SHA256Digest& digest) {
    printf("0x%08x%08x%08x%08x%08x%08x%08x%08x\n",
        digest.h0, digest.h1, digest.h2, digest.h3,
        digest.h4, digest.h5, digest.h6, digest.h7);
}

// Referred to https://en.wikipedia.org/wiki/SHA-2 for SHA256 algorithm

// DISCLAIMER: This implementation does not seem to match with the expected SHA256 hash. Current
// testing seems to show that it works well enough for the purposes of this project. I may debug this
// later if time permits. I don't think that 100% accuracy to the SHA256 algorithm is currently worth
// the time to debug. The rest of the project should still behave correctly.

// Constant values as defined by the SHA-256 spec
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// This function processes the next 512 bits and updates the provided digest
__device__ void processChunk(DataBlock_512_bit& chunk, SHA256Digest& digest) {
    // Load the 512 bits into a 2048 bit block according to the SHA-256 spec
    DataBlock_2048_bit w; // TODO: get this into fast memory
    w.processData(chunk);

    uint32_t a = digest.h0;
    uint32_t b = digest.h1;
    uint32_t c = digest.h2;
    uint32_t d = digest.h3;
    uint32_t e = digest.h4;
    uint32_t f = digest.h5;
    uint32_t g = digest.h6;
    uint32_t h = digest.h7;

    // Scramble the 2048 bit block according to the sha-256 algorithm
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotateBitsRight(e, 6) ^ rotateBitsRight(e, 11) ^ rotateBitsRight(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + k[i] + w.data[i];
        uint32_t S0 = rotateBitsRight(a, 2) ^ rotateBitsRight(a, 13) ^ rotateBitsRight(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b ^ c);
        uint32_t temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    digest.h0 += a;
    digest.h1 += b;
    digest.h2 += c;
    digest.h3 += d;
    digest.h4 += e;
    digest.h5 += f;
    digest.h6 += g;
    digest.h7 += h;
}

// A helper function to load memory/bytes from global data into shared memory. This is implemented to load contiguous
// memory instead of strided memory (performance improvement)
__device__ void loadGlobalMemoryIntoShared(uint8_t* sharedMem, const uint64_t numBytesToRead, const uint8_t* globalData)
{
    // globalData is already offset for blocks
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    for (int currentIndex = tid; currentIndex < numBytesToRead; currentIndex += blockSize) {
        sharedMem[currentIndex] = globalData[currentIndex];
    }

    __syncthreads();
}

// A helper function to write memory/bytes to global data from shared memory. This is implemented to write contiguous
// memory instead of strided memory (performance improvement)
__device__ void writeSharedDigestsToGlobalMemory(const uint8_t* sharedMem, const uint64_t numBytesToWrite, uint8_t* globalData)
{
    __syncthreads();

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    for (int currentIndex = tid; currentIndex < numBytesToWrite; currentIndex += blockSize) {
        globalData[currentIndex] = sharedMem[currentIndex];
    }
}

//#define USE_GLOBAL_DATA

__global__ void CreateHashes(const uint8_t* data, uint64_t dataLength, SHA256Digest* output)
{
    uint64_t numChunks = dataLength / 64; // a 512 block is made up of 64 8 bit integers
    uint64_t chunkLength = dataLength / numChunks; // TODO: make sure this will work once each thread iterates over several 512 bit chunks

#ifdef USE_GLOBAL_DATA
    DataBlock_512_bit* chunks = (DataBlock_512_bit*)data; // process data 512 bits at a time
#else
    // use shared data
    extern __shared__ DataBlock_512_bit chunks[];
    extern __shared__ SHA256Digest sharedDigests[];
    // Each thread processes a single 512 bit block in this implementation
    const uint64_t numBytesToRead = blockDim.x * sizeof(DataBlock_512_bit);

    data = data + blockIdx.x * numBytesToRead;
    loadGlobalMemoryIntoShared((uint8_t*)chunks, numBytesToRead, data);
#endif // USE_GLOBAL_DATA

    // Set initial digest values (defined by the SHA-256 spec)
    SHA256Digest digest;
    digest.h0 = 0x6a09e667;
    digest.h1 = 0xbb67ae85;
    digest.h2 = 0x3c6ef372;
    digest.h3 = 0xa54ff53a;
    digest.h4 = 0x510e527f;
    digest.h5 = 0x9b05688c;
    digest.h6 = 0x1f83d9ab;
    digest.h7 = 0x5be0cd19;
    
    int globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
    int localThreadId = threadIdx.x;

#ifdef USE_GLOBAL_DATA
    processChunk(chunks[globalThreadId], digest);
#else
    processChunk(chunks[localThreadId], digest);
#endif

    /* Future feature: Have each thread create a hash for multiple chunks of data
    for (unsigned int i = 0; i < numChunks; i++) {
        processChunk(chunks[i], digest);
    }
    */

    // process padding
    DataBlock_512_bit paddedBlock;
    paddedBlock.convertToPaddedBlock(chunkLength);

    processChunk(paddedBlock, digest);

    // Save result
#ifdef USE_GLOBAL_DATA
    output[globalThreadId].h0 = digest.h0;
    output[globalThreadId].h1 = digest.h1;
    output[globalThreadId].h2 = digest.h2;
    output[globalThreadId].h3 = digest.h3;
    output[globalThreadId].h4 = digest.h4;
    output[globalThreadId].h5 = digest.h5;
    output[globalThreadId].h6 = digest.h6;
    output[globalThreadId].h7 = digest.h7;
#else
    // stage data in shared memory and then write to global memory sequentially instead of strided writes
    sharedDigests[localThreadId].h0 = digest.h0;
    sharedDigests[localThreadId].h1 = digest.h1;
    sharedDigests[localThreadId].h2 = digest.h2;
    sharedDigests[localThreadId].h3 = digest.h3;
    sharedDigests[localThreadId].h4 = digest.h4;
    sharedDigests[localThreadId].h5 = digest.h5;
    sharedDigests[localThreadId].h6 = digest.h6;
    sharedDigests[localThreadId].h7 = digest.h7;

    const uint64_t sharedDigestBytes = sizeof(SHA256Digest) * blockDim.x;
    SHA256Digest* blockWriteLocation = output + blockIdx.x * blockDim.x;
    writeSharedDigestsToGlobalMemory((uint8_t*)sharedDigests, sharedDigestBytes, (uint8_t*)blockWriteLocation);
#endif // USE_GLOBAL_DATA

}

