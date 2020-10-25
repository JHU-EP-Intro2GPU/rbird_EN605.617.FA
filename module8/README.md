Build the assignment:
make

The make rule outputs two executables:
montecarlo.exe
    built from montecarlo.cu, kernels.cu, assignment.h, and kernels.h
    
    * This application uses random data to estimate the value of PI
    
matrixMultiply.exe
    build from matrixMultiply.cu and assignment.h
    
    * This application uses cublas version 2 to do matrix multiplication
      for single and double precision values


Example run:
## Matrix Multiply ##
# Run default values
./matrixMultiply.exe

# Run class example. Print matrices
./matrixMultiply --A 2 --B 9 --C 2 --debug

# Large
./matrixMultiply --A 4096 --B 4096 --C 4096

# Large with random values
./matrixMultiply --A 4096 --B 4096 --C 4096 --random


## Monte Carlo PI estimate ##
# Default arguments
./montecarlo.exe


# Print test data
./montecarlo.exe --debug

# Test random data
./montecarlo.exe --debug --random


# Test large data
--elements 100000 --blocksize 64 --random


# Test LARGE data
--elements 1000000 --blocksize 64 --random
Monte Carlo Algorithm Execution time = 5 seconds 980 milliseconds 311 microseconds
Total points within circle: 785661
pi estimate: 3.142644