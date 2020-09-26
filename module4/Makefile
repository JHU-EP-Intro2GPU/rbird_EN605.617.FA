all: assignment.c
	gcc assignment.c -o assignment.exe

host: host_memory.cu
	nvcc host_memory.cu -L /usr/local/cuda/lib -lcudart -o host_memory

global: global_memory.cu
	nvcc global_memory.cu -lcudart -o global_memory
