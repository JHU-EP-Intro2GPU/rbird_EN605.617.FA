#pragma once
#ifndef SAMPLE_TEST_DATA_H
#define SAMPLE_TEST_DATA_H

#include "CudaHelper.h"

HostAndDeviceMemory<uint8_t> readDataOneChunkOneIteration() {
    // Sample data
    uint64_t fileSizeInBytes = 64; // allocate 512 bits (1 chunk). 64 * 8 bits = 512 bits
    HostAndDeviceMemory<uint8_t> fileData;
    fileData.allocate(fileSizeInBytes);

    std::printf("bytes: %d\n", fileSizeInBytes);
    for (int i = 0; i < fileSizeInBytes; i++) {
        //fileData.host()[i] = 0;
        fileData.host()[i] = (i == 0) ? 'b' : 'a';
        std::printf("%c", fileData.host()[i]);
    }
    std::printf("\n");

    fileData.transferToDevice();
    return fileData;
}


HostAndDeviceMemory<uint8_t> readData2Chunks() {
    // Sample data
    const uint64_t fileSizeInBytes = 128; // allocate 2 blocks/chunks (1024 bits). 128 * 8 bits = 1024 bits
    HostAndDeviceMemory<uint8_t> fileData;
    fileData.allocate(fileSizeInBytes);

    std::printf("bytes: %d\n", fileSizeInBytes);
    int i = 0;

    // first chunk: "a" x 64
    for (; i < fileSizeInBytes / 2; i++) {
        fileData.host()[i] = 'a';
        std::printf("%c", fileData.host()[i]);
    }
    std::printf("\n");

    // second chunk: "b" + "a" x 63
    for (; i < fileSizeInBytes; i++) {
        //fileData.host()[i] = 0;
        fileData.host()[i] = (i == fileSizeInBytes / 2) ? 'b' : 'a';
        std::printf("%c", fileData.host()[i]);
    }
    std::printf("\n");

    fileData.transferToDevice();
    return fileData;
}

#endif // !SAMPLE_TEST_DATA_H

