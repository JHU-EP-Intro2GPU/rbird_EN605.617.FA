# Build standard hello world
g++ -o hello_world helloworld.cpp

# Build Nvidia code
nvcc -o hello_nvidia hello-world.cu

# Build opencl hello world
g++ -o hello_open_cl hello_world_cl.c -lOpenCL

# Build opencl device query
g++ -o clDeviceQuery clDeviceQuery.cpp -lOpenCL

