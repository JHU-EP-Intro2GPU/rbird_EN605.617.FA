
#include <cstdint>
#include <iostream>
#include <fstream>

#include "CudaHelper.h"
#include "CudaMerkleTree.h"

#include "SampleTestData.h"

#pragma region CommandLineArguments

struct CommandLineParameters
{
public:
    CommandLineParameters(int argc, const char** argv) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--outfile") == 0) {
                outfile = argv[++i];
            }
            else if (strcmp(arg, "--debug") == 0) {
                debug = true;
            }
            else if (strcmp(arg, "--randomBytes") == 0) {
                runLargeTest = true;
                totalRandomBytes = atoll(argv[++i]);
            }
        }
    }

    std::string outfile;
    bool debug = false;
    bool runLargeTest = false;
    uint64_t totalRandomBytes = 0;
};

#pragma endregion

#pragma region FileWriting
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

#pragma endregion

HostAndDeviceMemory<SHA256Digest> runTest(const HostAndDeviceMemory<uint8_t>& fileData, int blocks, int threadsPerBlock, bool printMessages)
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

// A helper function to convert/cast HostAndDeviceMemory from one type to another
// Example:
// Convert HostAndDeviceMemory<SHA256Digest> to byte buffer HostAndDeviceMemory<uint8_t>
class Conversion
{
public:
    template <typename T, typename S>
    static HostAndDeviceMemory<S> convertTo(HostAndDeviceMemory<T>& src) {
        HostAndDeviceMemory<S> other;

        other.host_ptr = (S*)src.host_ptr;
        other.device_ptr = (S*)src.device_ptr;

        // Calculate the new number of elements
        float sizeChange = ((float)sizeof(S)) / sizeof(T);
        other._size = src._size / sizeChange;

        // Clear out the source memory to avoid double-free
        src.host_ptr = nullptr;
        src.device_ptr = nullptr;
        src._size = 0;
        src._count = 0;

        return other;
    }
};

void runLargeTest(const CommandLineParameters& args);

int main(int argc, const char* argv[]) {
    CommandLineParameters args(argc, argv);
    std::ofstream outfile;

    if (!args.outfile.empty()) {
        outfile.open(args.outfile.c_str());
    }

    if (args.runLargeTest) {
        // The large test has special handling that doesn't match the tree test below.
        // Call a helper function to run test and return
        runLargeTest(args);
        return 0;
    }

    /* Iterative testing
    runTest(readDataOneChunkOneIteration(), 1, 1, args.debug);
    runTest(readData2Chunks(), 1, 2, args.debug);
    runTest(readData2Chunks(), 2, 1, args.debug);

    runTest(readData8Chunks(), 1, 8, args.debug); // 1 block, 8 threads
    runTest(readData8Chunks(), 2, 4, args.debug); // 1 block, 8 threads
    runTest(readData8Chunks(), 4, 2, args.debug); // 1 block, 8 threads
    */

    // test creating tree
    HostAndDeviceMemory<SHA256Digest> results = runTest(readData8Chunks(), 8, 1, args.debug);
    if (outfile.is_open()) {
        writeResults(results, outfile);
    }

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

void runLargeTest(const CommandLineParameters& args)
{
    uint64_t totalBytes = args.totalRandomBytes;

    // to avoid needing to do SHA256 padding, up-scale the total bytes
    // so that the chunks can be evenly divided into 512 bit chunks
    // NOTE: this adjustment is not perfect because the kernel code isn't perfect with 
    auto remainder = totalBytes % bytesPerBlock;
    if (remainder != 0) {
        totalBytes = totalBytes + (bytesPerBlock - remainder);
        std::cout << "Adjusting size to " << totalBytes << " bytes to be evenly divisible by " << bytesPerBlock << std::endl;
    }

    auto bytes = populateRandomData(totalBytes);

    if (args.debug) {
        if (totalBytes <= 5000) {
            std::cout << "Data:" << std::endl << bytes << std::endl;
        }
    }

    {
        TimeCodeBlockCuda totalRun("Merkle Tree Creation");
        HostAndDeviceMemory<SHA256Digest> results;
        size_t numDigests;
        do {
            // Note: these calculations are not perfect because the kernel does not have proper bounds checking.
            //       we are occasionally being short 1 block if the data length doesn't match up properly.
            const auto numberOfDataChunks = totalBytes / bytesPerBlock;

            // Calculate the number of threads per block. Max out at 'bytesPerBlock' as that is the maximum
            // number of threads that can load data from global memory (each thread loads 1 byte each into main memory)
            int threadsPerBlock = std::min(numberOfDataChunks, (uint64_t)bytesPerBlock);
            int totalBlocks = std::max(numberOfDataChunks / threadsPerBlock, (uint64_t)1);

            if (args.debug) {
                std::cout << "Bytes: " << totalBytes << std::endl;
                std::printf("Blocks: %d ThreadsPerBlock: %d\n", totalBlocks, threadsPerBlock);
            }

            // Do not print debug data for large datasets
            results = runTest(bytes, totalBlocks, threadsPerBlock, false);
            numDigests = results.size();

            // prepare the results to be the next input data
            totalBytes = results.size() * sizeof(SHA256Digest);
            bytes = Conversion::convertTo<SHA256Digest, uint8_t>(results);
        } while (numDigests > 1);

    }
}
