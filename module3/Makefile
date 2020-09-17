all: assignment.c
	gcc assignment.c -o assignment.exe

block:
	nvcc blocks.cu -o blocks_default

	nvcc blocks.cu -D ARRAY_SIZE=256 -D BLOCK_SIZE=32 -o blocks_256_32
	nvcc blocks.cu -D ARRAY_SIZE=256 -D BLOCK_SIZE=64 -o blocks_256_64

	nvcc blocks.cu -D ARRAY_SIZE=512 -D BLOCK_SIZE=20 -o blocks_512_20
	nvcc blocks.cu -D ARRAY_SIZE=512 -D BLOCK_SIZE=48 -o blocks_512_48

	nvcc blocks.cu -D ARRAY_SIZE=512 -D BLOCK_SIZE=128 -o blocks_512_128
	nvcc blocks.cu -D ARRAY_SIZE=1000 -D BLOCK_SIZE=96 -o blocks_1000_96

grid:
	nvcc grids.cu -o grids

