
#include <stdio.h>
#include <iostream>
#include <cstdint>

#include "CudaHelper.h"
#include "CudaMerkleTree.h"

void printDigest(const SHA256Digest& digest) {
    std::cout << "0x" << std::hex
        << digest.h0 << digest.h1 << digest.h2 << digest.h3
        << digest.h4 << digest.h5 << digest.h6 << digest.h7;
    std::cout << std::endl;
}

int main(int argc, const char* argv[]) {

    std::printf("Hello world!\n");

    uint64_t fileSizeInBytes = 64; // allocate 512 bits (1 chunk). 64 * 8 bits = 512 bits
    HostAndDeviceMemory<uint8_t> fileData;
    fileData.allocate(fileSizeInBytes);

    for (int i = 0; i < fileSizeInBytes; i++) {
        fileData.host()[i] = 0;
    }

    fileData.transferToDevice();

    HostAndDeviceMemory<SHA256Digest> messageDigest(1); // allocate 1 digest

    int blocks = 1;
    int threadsPerBlock = 1;
    CreateHashes <<< blocks, threadsPerBlock >>> (fileData.device(), fileSizeInBytes, messageDigest.device());
    gpuErrchk(cudaGetLastError());

    messageDigest.transferToHost();

    printDigest(messageDigest.host()[0]);


    // this app can enforce an exact file size restriction in order to not deal with
    // special padding on the final chunk

    return 0;
}

