all: assignment.c
	gcc assignment.c -o assignment.exe

host: host_memory.cu
	nvcc host_memory.cu -std=c++11 -L /usr/local/cuda/lib -lcudart -o host_memory

global: global_memory.cu
	nvcc global_memory.cu -std=c++11 -lcudart -o global_memory
