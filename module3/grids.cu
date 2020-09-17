#include <stdio.h>

__global__ void what_is_my_id_2d_A(
	unsigned int* const block_x,
	unsigned int* const block_y,
	unsigned int* const thread,
	unsigned int* const calc_thread,
	unsigned int* const x_thread,
	unsigned int* const y_thread,
	unsigned int* const grid_dimx,
	unsigned int* const block_dimx,
	unsigned int* const grid_dimy,
	unsigned int* const block_dimy)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread[thread_idx] = threadIdx.x;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;
}

#ifndef ARRAY_SIZE_X
#define ARRAY_SIZE_X 32
#endif // !ARRAY_SIZE_X

#ifndef ARRAY_SIZE_Y
#define ARRAY_SIZE_Y 16
#endif // !ARRAY_SIZE_Y


#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(unsigned int)))

/* Declare statically six arrays of ARRAY_SIZE each */
unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_warp[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

int main(void)
{
	/* Total thread count = 32 * 4 = 128 */
	const dim3 threads_rect(32, 4);
	const dim3 blocks_rect(1, 4);

	/* Total thread count = 16 * 8 = 128 */
	const dim3 threads_square(16, 8); /* 16 * 8 */
	const dim3 blocks_square(2, 2);

	/* Total thread count = 8 * 16 = 128 */
	const dim3 reverse_threads_square(8, 16); /* 8 * 16 */
	const dim3 reverse_blocks_square(2, 2);

	/* Total thread count = 4 * 32 = 128 */
	const dim3 reverse_threads_rect(4, 32); /* 4 * 32 */
	const dim3 reverse_blocks_rect(4, 1);

	/* Total thread count = 4 * 2 = 8 */
	const dim3 few_threads_rect(4, 2); /* 4 * 2 */
	const dim3 many_blocks_rect(8, 2);


	/* Total thread count = 4 * 2 = 8 */
	const dim3 many_threads(1, 512); /* 4 * 2 */
	const dim3 one_blocks_rect(1, 1);


	/* Needed to wait for a character at exit */
	char ch;

	/* Declare statically six arrays of ARRAY_SIZE each */
	unsigned int* gpu_block_x;
	unsigned int* gpu_block_y;
	unsigned int* gpu_thread;
	unsigned int* gpu_warp;
	unsigned int* gpu_calc_thread;
	unsigned int* gpu_xthread;
	unsigned int* gpu_ythread;
	unsigned int* gpu_grid_dimx;
	unsigned int* gpu_block_dimx;
	unsigned int* gpu_grid_dimy;
	unsigned int* gpu_block_dimy;

	/* Allocate arrays on the GPU */
	cudaMalloc((void**)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_warp, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_xthread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_ythread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);

	for (int kernel = 0; kernel < 6; kernel++)
	{
		dim3 threads_dim;
		dim3 blocks_dim;
		switch (kernel)
		{
		case 0:
		{
			blocks_dim = blocks_rect;
			threads_dim = threads_rect;
			/* Execute our kernel */
			what_is_my_id_2d_A << <blocks_rect, threads_rect >> > (gpu_block_x, gpu_block_y,
				gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
				gpu_grid_dimy, gpu_block_dimy);
		} break;

		case 1:
		{
			blocks_dim = blocks_square;
			threads_dim = threads_square;

			/* Execute our kernel */
			what_is_my_id_2d_A << <blocks_square, threads_square >> > (gpu_block_x, gpu_block_y,
				gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
				gpu_grid_dimy, gpu_block_dimy);
		} break;
		case 2:
		{
			blocks_dim = reverse_blocks_square;
			threads_dim = reverse_threads_square;

			/* Execute our kernel */
			what_is_my_id_2d_A << <reverse_blocks_square, reverse_threads_square >> > (gpu_block_x, gpu_block_y,
				gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
				gpu_grid_dimy, gpu_block_dimy);
		} break;
		case 3:
		{
			blocks_dim = reverse_blocks_square;
			threads_dim = reverse_threads_rect;

			/* Execute our kernel */
			what_is_my_id_2d_A << <reverse_blocks_square, reverse_threads_rect >> > (gpu_block_x, gpu_block_y,
				gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
				gpu_grid_dimy, gpu_block_dimy);
		} break;
		case 4:
		{
			blocks_dim = many_blocks_rect;
			threads_dim = few_threads_rect;

			/* Execute our kernel */
			what_is_my_id_2d_A << <many_blocks_rect, few_threads_rect >> > (gpu_block_x, gpu_block_y,
				gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
				gpu_grid_dimy, gpu_block_dimy);
		} break;
		case 5:
		{
			blocks_dim = many_threads;
			threads_dim = one_blocks_rect;

			/* Execute our kernel */
			what_is_my_id_2d_A << <many_threads, one_blocks_rect >> > (gpu_block_x, gpu_block_y,
				gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
				gpu_grid_dimy, gpu_block_dimy);
		} break;


		default: exit(1); break;
		}

		/* Copy back the gpu results to the CPU */
		cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_xthread, gpu_xthread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_ythread, gpu_ythread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

		printf("\nKernel %d\n", kernel);
		printf("Block Dimensions: x= %2d, y=%2d, z=%2d\n", blocks_dim.x, blocks_dim.y, blocks_dim.z);
		printf("Thread Dimensions: x= %2d, y=%2d, z=%2d\n", threads_dim.x, threads_dim.y, threads_dim.z);
		/* Iterate through the arrays and print */
		for (int y = 0; y < ARRAY_SIZE_Y; y++)
		{
			for (int x = 0; x < ARRAY_SIZE_X; x++)
			{
				printf("CT: %3u BKX: %2u BKY: %2u TID: %3u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY: %1u BDY: %1u\n",
					cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x], cpu_thread[y][x], cpu_ythread[y][x],
					cpu_xthread[y][x], cpu_grid_dimx[y][x], cpu_block_dimx[y][x], cpu_grid_dimy[y][x], cpu_block_dimy[y][x]);

			}
		}


	}

	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread);
	cudaFree(gpu_warp);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_xthread);
	cudaFree(gpu_ythread);
	cudaFree(gpu_grid_dimy);
	cudaFree(gpu_block_dimy);
}
