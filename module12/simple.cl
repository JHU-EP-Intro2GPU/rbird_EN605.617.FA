//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void square(__global int* buffer)
{
	size_t id = get_global_id(0);
	printf("%d: %d\n", id, buffer[id]);
	buffer[id] = buffer[id] * buffer[id];
}

__kernel void average2D(__global int* buffer2D, int stride, __local int* sharedMem)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	
	size_t bufferIndex = y * stride + x;
	size_t localIndex = x * get_global_size(0) + y;

	int value = buffer2D[bufferIndex];

	printf("(%d, %d)[%d] = %d, local index: %d\n", x, y, bufferIndex, value, localIndex);
	
	// sum values in local memory
	sharedMem[localIndex] = value;

	// perform a reduction
	size_t localGroupSize = get_local_size(0) * get_local_size(1);
	int offset = localGroupSize / 2;
	while (offset > 0) {
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if (localIndex < offset) {
			sharedMem[localIndex] += sharedMem[localIndex + offset];
		}

		offset /= 2;
	}

	if (localIndex == 0) {
		printf("Sum: %d\n", sharedMem[0]);
		printf("Average: %f\n", ((float) sharedMem[0]) / localGroupSize);
	}
}
