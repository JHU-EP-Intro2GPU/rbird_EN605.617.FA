
#include <cstdint>

#include "CudaHelper.h"
#include "CudaMerkleTree.h"

#include "SampleTestData.h"



int main(int argc, const char* argv[]) {
    HostAndDeviceMemory<uint8_t> fileData = readData2Chunks();


    HostAndDeviceMemory<SHA256Digest> messageDigest(2); // allocate 1 digest

    int blocks = 2;
    int threadsPerBlock = 1; // TODO: this is not accurate long term
    CreateHashes <<< blocks, threadsPerBlock >>> (fileData.device(), fileData.size(), messageDigest.device());
    gpuErrchk(cudaGetLastError());

    messageDigest.transferToHost();

    for (int i = 0; i < messageDigest.size(); i++)
        printDigest(messageDigest.host()[i]);


    // this app can enforce an exact file size restriction in order to not deal with
    // special padding on the final chunk

    return 0;
}

