all: assignment.cu GlobalKernels.cu RegisterKernels.cu SharedKernels.cu
	nvcc assignment.cu GlobalKernels.cu RegisterKernels.cu SharedKernels.cu -std=c++11 -lcudart -o assignment.exe

register: register.cu
	nvcc register.cu -std=c++11 -lcudart -o register
