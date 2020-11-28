
#include <cstdint>

#include "CudaHelper.h"
#include "CudaMerkleTree.h"

#include "SampleTestData.h"


void runTest(HostAndDeviceMemory<uint8_t>& fileData, int blocks, int threadsPerBlock)
{
    std::cout << "Data Bytes: " << fileData.size() << ", Blocks: " << blocks << ", Threads Per Block: " << threadsPerBlock << std::endl;
    std::cout << "Data:" << std::endl << fileData << std::endl;

    if (fileData.size() % bytesPerBlock != 0) {
        std::cerr << "Unexpected data size: " << fileData.size() << std::endl;
        exit(-1);
    }

    // Current implementation, 1 hash per thread
    size_t totalChunks = blocks * threadsPerBlock;
    HostAndDeviceMemory<SHA256Digest> messageDigest(totalChunks);

    // Allocate enough shared memory to store 1 file chunk
    size_t sharedMemoryBytes = threadsPerBlock * bytesPerBlock;
    CreateHashes <<< blocks, threadsPerBlock, sharedMemoryBytes >>> (fileData.device(), fileData.size(), messageDigest.device());
    gpuErrchk(cudaGetLastError());

    messageDigest.transferToHost();

    std::printf("\nHashes:\n");
    std::printf("Hash bytes: %d\n", messageDigest.size() * sizeof(SHA256Digest));
    for (int i = 0; i < messageDigest.size(); i++)
        printDigest(messageDigest.host()[i]);

    std::printf("\n");
}


int main(int argc, const char* argv[]) {
    runTest(readDataOneChunkOneIteration(), 1, 1);
    runTest(readData2Chunks(), 1, 2);
    runTest(readData2Chunks(), 2, 1);

    runTest(readData8Chunks(), 1, 8); // 1 block, 8 threads
    runTest(readData8Chunks(), 2, 4); // 1 block, 8 threads
    runTest(readData8Chunks(), 4, 2); // 1 block, 8 threads

    runTest(readData8Chunks(), 8, 1); // 1 block, 8 threads


    // this app can enforce an exact file size restriction in order to not deal with
    // special padding on the final chunk

    return 0;
}

