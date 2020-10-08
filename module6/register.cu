#include <stdio.h>
#include <stdlib.h>

#include "assignment.h"

// Don't get too fancy with these numbers
//#define NUM_ELEMENTS 128
//#define THREADS_PER_BLOCK 32



// 109 microseconds
//#define NUM_THREADS 128
//#define LOOP_UNROLL 1

// fastest: 96, times varied quite a bit though
//#define NUM_THREADS 64
//#define LOOP_UNROLL 2


#define THREADS_PER_BLOCK 128
#define NUM_ELEMENTS (THREADS_PER_BLOCK * 100000)

// messing around with the LOOP_UNROLL requires actual changes
// to the kernel to represent an accurate test
#define LOOP_UNROLL 8
#define NUM_THREADS (NUM_ELEMENTS / LOOP_UNROLL)


static_assert(LOOP_UNROLL* NUM_THREADS == NUM_ELEMENTS, "Loop Unroll does not line up with the number of elements");
static_assert(NUM_ELEMENTS % THREADS_PER_BLOCK == 0, "There is not an equal division of elements among the blocks");

__host__ void wait_exit(void)
{
        char ch;

        printf("\nPress any key to exit");
        ch = getchar();
}

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
        for(unsigned int i=0; i < NUM_ELEMENTS; i++)
        {
                host_data_ptr[i] = (unsigned int) rand();
        }
}

__global__ void test_gpu_register(unsigned int * const data, const unsigned int num_elements)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        unsigned int d_tmp = data[tid];
        unsigned int d_tmp2 = data[tid + NUM_THREADS];
        unsigned int d_tmp3 = data[tid + NUM_THREADS * 2];
        unsigned int d_tmp4 = data[tid + NUM_THREADS * 3];
        unsigned int d_tmp5 = data[tid + NUM_THREADS * 4];
        unsigned int d_tmp6 = data[tid + NUM_THREADS * 5];
        unsigned int d_tmp7 = data[tid + NUM_THREADS * 6];
        unsigned int d_tmp8 = data[tid + NUM_THREADS * 7];

        d_tmp = d_tmp * 2;
        d_tmp2 = d_tmp2 * 2;
        d_tmp3 = d_tmp3 * 2;
        d_tmp4 = d_tmp4 * 2;
        d_tmp5 = d_tmp5 * 2;
        d_tmp6 = d_tmp6 * 2;
        d_tmp7 = d_tmp7 * 2;
        d_tmp8 = d_tmp8 * 2;

        data[tid] = d_tmp;
        data[tid + NUM_THREADS] = d_tmp2;
        data[tid + NUM_THREADS * 2] = d_tmp3;
        data[tid + NUM_THREADS * 3] = d_tmp4;
        data[tid + NUM_THREADS * 4] = d_tmp5;
        data[tid + NUM_THREADS * 5] = d_tmp6;
        data[tid + NUM_THREADS * 6] = d_tmp7;
        data[tid + NUM_THREADS * 7] = d_tmp8;

//        for (int globalIndex = tid; globalIndex < num_elements; globalIndex += (NUM_THREADS * LOOP_UNROLL)) {
//        }
}

__global__ void test_gpu_register_array(unsigned int* const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Fun test, but doesn't seem to use registers
    unsigned int values[LOOP_UNROLL];

    // Load
    for (int i = 0; i < LOOP_UNROLL; i++) {
        values[i] = data[tid + (NUM_THREADS * i)];
    }

    // Calculate
    for (int i = 0; i < LOOP_UNROLL; i++) {
        values[i] = 2 * values[i];
    }

    // Write
    for (int i = 0; i < LOOP_UNROLL; i++) {
         data[tid + (NUM_THREADS * i)] = values[i];
    }
}

__host__ void gpu_kernel(void)
{
        const unsigned int num_elements = NUM_ELEMENTS;
        const unsigned int num_threads = NUM_THREADS;
        const unsigned int num_blocks = (NUM_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        const unsigned int num_bytes = num_elements * sizeof(unsigned int);

        printf("Elements: %d Threads: %d BlockSize: %d LoopUnroll: %d\n", NUM_ELEMENTS, NUM_THREADS, THREADS_PER_BLOCK, LOOP_UNROLL);

        unsigned int * data_gpu;

        unsigned int * host_packed_array = new unsigned int[num_elements];
        unsigned int * host_packed_array_output = new unsigned int[num_elements];

        cudaMalloc(&data_gpu, num_bytes);

        generate_rand_data(host_packed_array);

        cudaMemcpy(data_gpu, host_packed_array, num_bytes,cudaMemcpyHostToDevice);

        {
            TimeCodeBlock kernelTime("Kernel runtime");
            test_gpu_register <<<num_blocks, THREADS_PER_BLOCK >>>(data_gpu, num_elements);

            //test_gpu_register_array <<<num_blocks, THREADS_PER_BLOCK >>> (data_gpu, num_elements);

            cudaThreadSynchronize();        // Wait for the GPU launched work to complete
        }
        cudaGetLastError();

        cudaMemcpy(host_packed_array_output, data_gpu, num_bytes,cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_elements; i++){
                //printf("Input value: %x, device output: %x\n",host_packed_array[i], host_packed_array_output[i]);

                if (host_packed_array[i] * 2 != host_packed_array_output[i]) {
                    printf("ERROR at index %d: %d %d\n", i, host_packed_array[i], host_packed_array_output[i]);
                }
        }

        cudaFree((void* ) data_gpu);
        cudaDeviceReset();

        delete[] host_packed_array;
        delete[] host_packed_array_output;

        wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	gpu_kernel();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return EXIT_SUCCESS;
}
