#pragma once
#ifndef SAMPLE_TEST_DATA_H
#define SAMPLE_TEST_DATA_H

#include "CudaHelper.h"

#include <string>
#include <iostream>

// 512 bits per 1 chunk. 64 bytes * 8 bits = 512 bits
constexpr unsigned int bytesPerBlock = 64;

// test block data whose expected hashes are known (note: my hash isn't correct to the exact sha 256 standard)
const std::string aaa_block = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const std::string baa_block = "baaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

const std::string aaa_expected_hash = "0x1f2e53d352d0e7000b7d985ab848e3b0f59b7726c86d8b8c408b00de5adf2d9b";
const std::string baa_expected_hash = "0xd24bfb00f5be022baddb4da7806f9c2eb278b7a7c47a6792bd9eb75b858daade";

HostAndDeviceMemory<uint8_t>& operator<<(HostAndDeviceMemory<uint8_t>& data, const std::string& dataToAdd) {
    for (char c : dataToAdd) {
        data.appendToHost(c);
    }

    return data;
}

std::ostream& operator<<(std::ostream& out, const HostAndDeviceMemory<uint8_t>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        // print a newline at each chunk border
        if (i != 0 && ((i % bytesPerBlock) == 0)) {
            out << std::endl;
        }

        out << data.host()[i];
    }

    return out;
}


// Data is used to verify correct behavior from the hasing code
HostAndDeviceMemory<uint8_t> readDataOneChunkOneIteration() {
    uint64_t fileSizeInBytes = bytesPerBlock;
    HostAndDeviceMemory<uint8_t> fileData(fileSizeInBytes);

    fileData << aaa_block;
//    fileData << baa_block;

    fileData.transferToDevice();
    return fileData;
}

// Test is used to verify correct behavior between threads and blocks
HostAndDeviceMemory<uint8_t> readData2Chunks() {
    // Sample data
    const uint64_t fileSizeInBytes = bytesPerBlock * 2;
    HostAndDeviceMemory<uint8_t> fileData(fileSizeInBytes);

    fileData << aaa_block << baa_block;

    fileData.transferToDevice();
    return fileData;
}

// Data used to test creating the hash tree
HostAndDeviceMemory<uint8_t> readData8Chunks() {
    const uint64_t fileSizeInBytes = bytesPerBlock * 8;
    HostAndDeviceMemory<uint8_t> fileData(fileSizeInBytes);

    fileData << aaa_block << baa_block << aaa_block << baa_block
        << aaa_block << baa_block << aaa_block << baa_block;

    fileData.transferToDevice();
    return fileData;
}

HostAndDeviceMemory<uint8_t> populateRandomData(const uint64_t dataSizeInBytes) {
    HostAndDeviceMemory<uint8_t> fileData(dataSizeInBytes);

    // TODO: This could be popuulated by curand
    for (int byteCount = 0; byteCount < dataSizeInBytes; byteCount++) {
        fileData.host()[byteCount] = (uint8_t)rand();
    }

    fileData.transferToDevice();
    return fileData;
}


#endif // !SAMPLE_TEST_DATA_H

