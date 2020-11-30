
#include <cstdint>

#include "CudaHelper.h"
#include "CudaMerkleTree.h"

#include "SampleTestData.h"


HostAndDeviceMemory<SHA256Digest> runTest(HostAndDeviceMemory<uint8_t>& fileData, int blocks, int threadsPerBlock)
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

    return messageDigest;
}

class Conversion
{
public:
    template <typename T, typename S>
    static HostAndDeviceMemory<S> convertTo(HostAndDeviceMemory<T>& src) {
        HostAndDeviceMemory<S> other;

        other.host_ptr = (S*)src.host_ptr;
        other.device_ptr = (S*)src.device_ptr;
        float sizeChange = ((float)sizeof(S)) / sizeof(T);
        other._size = src._size / sizeChange;

        src.host_ptr = nullptr;
        src.device_ptr = nullptr;
        src._size = 0;
        src._count = 0;

        return other;
    }
};

int main(int argc, const char* argv[]) {
    /*
    runTest(readDataOneChunkOneIteration(), 1, 1);
    runTest(readData2Chunks(), 1, 2);
    runTest(readData2Chunks(), 2, 1);

    runTest(readData8Chunks(), 1, 8); // 1 block, 8 threads
    runTest(readData8Chunks(), 2, 4); // 1 block, 8 threads
    runTest(readData8Chunks(), 4, 2); // 1 block, 8 threads
    */

    // test creating tree
    HostAndDeviceMemory<SHA256Digest> results = runTest(readData8Chunks(), 8, 1);
    do {
        int expectedNumberOfChunks = results.size() / 2; // 512 bit chunk converted to 256 bit digest
        //HostAndDeviceMemory<uint8_t> nextBatch = results.convertTo<uint8_t>();
        HostAndDeviceMemory<uint8_t> nextBatch = Conversion::convertTo<SHA256Digest, uint8_t>(results);
        results = runTest(nextBatch, expectedNumberOfChunks, 1);
    } while (results.size() > 1);


    // this app can enforce an exact file size restriction in order to not deal with
    // special padding on the final chunk

    return 0;
}

