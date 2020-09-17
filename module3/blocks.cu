#include <stdio.h>
#include <math.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 256
#endif // !ARRAY_SIZE

#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif // !BLOCK_SIZE


/* Declare  statically two arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];

__global__
void what_is_my_id(unsigned int* block, unsigned int* thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub0()
{

	/* Declare pointers for GPU based params */
	unsigned int* gpu_block;
	unsigned int* gpu_thread;

	cudaMalloc((void**)&gpu_block, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	const unsigned int threads_per_block = BLOCK_SIZE;
	const unsigned int num_blocks = ceil((double) ARRAY_SIZE / threads_per_block);

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, threads_per_block>>>(gpu_block, gpu_thread);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("i: %4u, Thread: %2u - Block: %2u\n", i, cpu_thread[i], cpu_block[i]);
	}
}

int main()
{
	main_sub0();

	return EXIT_SUCCESS;
}
