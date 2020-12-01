
#include <cstdint>
#include <iostream>
#include <fstream>

#include "CudaHelper.h"
#include "CudaMerkleTree.h"

#include "SampleTestData.h"

struct CommandLineParameters
{
public:
    CommandLineParameters(int argc, const char** argv) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--file") == 0) {
                outfile = argv[++i];
            }
            else if (strcmp(arg, "--debug") == 0) {
                debug = true;
            }
        }
    }

    std::string outfile;
    bool debug = false;
};

std::ostream& operator<<(std::ostream& out, const SHA256Digest& digest) {
    out << std::hex << digest.h0 << digest.h1 << digest.h2 << digest.h3
        << digest.h4 << digest.h5 << digest.h6 << digest.h7;

    return out;
}

void writeResults(const HostAndDeviceMemory<SHA256Digest>& results, std::ostream& output) {
    for (int i = 0; i < results.size(); i++) {
        if (i != 0) {
            output << " ";
        }
        output << results.host()[i];
    }

    output << std::endl;
}

HostAndDeviceMemory<SHA256Digest> runTest(HostAndDeviceMemory<uint8_t>& fileData, int blocks, int threadsPerBlock, bool printMessages)
{
    if (printMessages) {
        std::cout << "Data Bytes: " << fileData.size() << ", Blocks: " << blocks << ", Threads Per Block: " << threadsPerBlock << std::endl;
        std::cout << "Data:" << std::endl << fileData << std::endl;
    }

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

    if (printMessages) {
        std::printf("\nHashes:\n");
        std::printf("Hash bytes: %d\n", messageDigest.size() * sizeof(SHA256Digest));
        for (int i = 0; i < messageDigest.size(); i++)
            printDigest(messageDigest.host()[i]);

        std::printf("\n");
    }

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
    CommandLineParameters args(argc, argv);
    std::ofstream outfile;

    if (!args.outfile.empty()) {
        outfile.open(args.outfile.c_str());
    }

    /*
    runTest(readDataOneChunkOneIteration(), 1, 1, args.debug);
    runTest(readData2Chunks(), 1, 2, args.debug);
    runTest(readData2Chunks(), 2, 1, args.debug);

    runTest(readData8Chunks(), 1, 8, args.debug); // 1 block, 8 threads
    runTest(readData8Chunks(), 2, 4, args.debug); // 1 block, 8 threads
    runTest(readData8Chunks(), 4, 2, args.debug); // 1 block, 8 threads
    */

    // test creating tree
    HostAndDeviceMemory<SHA256Digest> results = runTest(readData8Chunks(), 8, 1, args.debug);
    do {
        int expectedNumberOfChunks = results.size() / 2; // 512 bit chunk converted to 256 bit digest
        //HostAndDeviceMemory<uint8_t> nextBatch = results.convertTo<uint8_t>();
        HostAndDeviceMemory<uint8_t> nextBatch = Conversion::convertTo<SHA256Digest, uint8_t>(results);
        results = runTest(nextBatch, expectedNumberOfChunks, 1, args.debug);

        if (outfile.is_open()) {
            writeResults(results, outfile);
        }
    } while (results.size() > 1);


    // this app can enforce an exact file size restriction in order to not deal with
    // special padding on the final chunk

    return 0;
}

