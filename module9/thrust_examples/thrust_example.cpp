#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

#define print(cuda_vector)\
  for(int i = 0; i < cuda_vector.size(); i++)\
  {\
    std::cout << #cuda_vector <<"[" << i << "] = " << cuda_vector[i] << std::endl;\
  }


int main(void)
{
  // H has storage for 4 integers
  thrust::host_vector<int> H(4);

  // initialize individual elements
  H[0] = 14;
  H[1] = 20;
  H[2] = 38;
  H[3] = 46;

  // H.size() returns the size of vector H
  std::cout << "H has size " << H.size() << std::endl;

  // print contents of H
    print(H);

  // resize H
  H.resize(2);

  std::cout << "H now has size " << H.size() << std::endl;
    
    print(H);
    
   std::cout << std::endl;

  // Copy host_vector H to device_vector D1 (there appears to be a delay on the first
    // call to the device, likely setting up the cuda context)
    std::cout << "Copying to device (D0)" << std::endl;
  thrust::device_vector<int> D0 = H;

    std::cout << "Copying to device (D)" << std::endl;
  thrust::device_vector<int> D = H;
    
    print(D);

  // elements of D can be modified
    std::cout << "Direct device manipulation" << std::endl;
  D[0] = 99;
  D[1] = 88;

  // print contents of D
    print(D);

  // H and D are automatically destroyed when the function returns
  return 0;
}
