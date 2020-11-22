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

__kernel void average2D(__global float* buffer2D, int stride, __local float* sharedMem)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	
	size_t bufferIndex = x * stride + y;
	size_t localIndex = x * get_global_size(0) + y;

	int value = buffer2D[bufferIndex];

	// DEBUG statement that correctly outputs on Intel chip
	//printf("(%d, %d)[%d] = %d, local index: %d\n", x, y, bufferIndex, value, localIndex);
	
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

	// clear out old memory for print readability
	buffer2D[bufferIndex] = 0;

	if (localIndex == 0) {
		printf("Sum: %f\n", sharedMem[0]);
		printf("Average: %f\n\n", sharedMem[0] / localGroupSize);

		buffer2D[bufferIndex] = sharedMem[0] / localGroupSize;
	}
}
