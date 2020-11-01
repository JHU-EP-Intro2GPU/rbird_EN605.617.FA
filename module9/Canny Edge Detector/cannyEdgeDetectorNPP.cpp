/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include "assignment.h"


namespace npp
{
    void
    loadImage(const std::string &rFileName, ImageCPU_8u_C3 &rImage)
    {
        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());

        // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
        {
            eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
        }

        NPP_ASSERT(eFormat != FIF_UNKNOWN);
        // check that the plugin has reading capabilities ...
        FIBITMAP *pBitmap;

        if (FreeImage_FIFSupportsReading(eFormat))
        {
            pBitmap = FreeImage_Load(eFormat, rFileName.c_str());
        }

        NPP_ASSERT(pBitmap != 0);
        // make sure this is an 8-bit single channel image
        NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_RGB);
        NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 24);

        // create an ImageCPU to receive the loaded image data
        ImageCPU_8u_C3 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

        // Copy the FreeImage data into the new ImageCPU
        unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
        const Npp8u *pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) -1);
        Npp8u *pDstLine = oImage.data();
        unsigned int nDstPitch = oImage.pitch();

        for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, oImage.width() * (sizeof(Npp8u)* 3));
            pSrcLine -= nSrcPitch;
            pDstLine += nDstPitch;
        }

        // swap the user given image with our result image, effecively
        // moving our newly loaded image data into the user provided shell
        oImage.swap(rImage);
    }
}

inline int cudaDeviceInit(int argc, const char **argv) {
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
    exit(EXIT_FAILURE);
  }

  int dev = findCudaDevice(argc, argv);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name
            << std::endl;

  checkCudaErrors(cudaSetDevice(dev));

  return dev;
}

bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  try {
    std::string sFilename;
    char *filePath;

    cudaDeviceInit(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }
      
    bool isColorImage = false;
    if (checkCmdLineFlag(argc, (const char **)argv, "color")) {
        isColorImage = true;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    } else {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath) {
      sFilename = filePath;
    } else {
      sFilename = "Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good()) {
      std::cout << "cannyEdgeDetectionNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    } else {
      std::cout << "cannyEdgeDetectionNPP unable to open: <" << sFilename.data()
                << ">" << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0) {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos) {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_cannyEdgeDetection.pgm";

    if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }


    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc;
    TimeCodeBlockCuda wholeProcess("Entire process");

    if (isColorImage) {
        npp::ImageCPU_8u_C3 colorImage;
        npp::loadImage(sFilename, colorImage);
                
        npp::ImageNPP_8u_C3 deviceColorImage(colorImage);
        oDeviceSrc = npp::ImageNPP_8u_C1(colorImage.width(), colorImage.height());
        
        // convert color image to grayscale
        NppiSize colorImageROI = {(int)deviceColorImage.width(), (int)deviceColorImage.height()};
        
        TimeCodeBlockCuda convertToGrayscale("Convert to gray scale");
        NPP_CHECK_NPP(nppiRGBToGray_8u_C3C1R(
            deviceColorImage.data(), deviceColorImage.pitch(),
            oDeviceSrc.data(), oDeviceSrc.pitch(), colorImageROI
        ));
        
    }
    else {
        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        oDeviceSrc = npp::ImageNPP_8u_C1(oHostSrc);
    }
    

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    int nBufferSize = 0;
    Npp8u *pScratchBufferNPP = 0;

    // get necessary scratch buffer size and allocate that much device memory
    NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize));

    cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

    // now run the canny edge detection filter
    // Using nppiNormL2 will produce larger magnitude values allowing for finer
    // control of threshold values while nppiNormL1 will be slightly faster.
    // Also, selecting the sobel gradient filter allows up to a 5x5 kernel size
    // which can produce more precise results but is a bit slower. Commonly
    // nppiNormL2 and sobel gradient filter size of 3x3 are used. Canny
    // recommends that the high threshold value should be about 3 times the low
    // threshold value. The threshold range will depend on the range of
    // magnitude values that the sobel gradient filter generates for a
    // particular image.

    Npp16s nLowThreshold = 72;
    Npp16s nHighThreshold = 256;

    std::printf("Image size: %d x %d\n", oSrcSize.height, oSrcSize.width);
    if ((nBufferSize > 0) && (pScratchBufferNPP != 0)) {
        TimeCodeBlockCuda findEdges("Find edges");
        // If color image, convert to greyscale
        
      NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
          oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
          oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, NPP_FILTER_SOBEL,
          NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
          NPP_BORDER_REPLICATE, pScratchBufferNPP));
    }

    // free scratch buffer memory
    cudaFree(pScratchBufferNPP);

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

  } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
