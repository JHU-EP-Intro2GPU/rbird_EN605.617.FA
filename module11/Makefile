all: assignment.c
	gcc assignment.c -o assignment.exe

convolution:
	nvcc Convolution.cpp -lOpenCL -o Convolution

info:
	nvcc OpenCLInfo.cpp -lOpenCL -o OpenCLInfo

vector: vector_example.cpp
	nvcc vector_example.cpp -lOpenCL -o vector_example