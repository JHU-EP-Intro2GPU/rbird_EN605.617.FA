//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Convolution.cl
//
//    This is a simple kernel performing convolution.


// mask radius distance for 7x7 mask
// 4 4 4 4 4 4 4
// 4 3 3 3 3 3 4
// 4 3 2 2 2 3 4
// 4 3 2 1 2 3 4
// 4 3 2 2 2 3 4
// 4 3 3 3 3 3 4
// 4 4 4 4 4 4 4

// all threads are calulating the same r & c values at the same time.
// warp thrashing should not be an issue for conditionals
float gradientValue(int r, int c, int maskHeight, int maskWidth) {
	int maskMidRow = maskHeight / 2;
	int maskMidCol = maskWidth / 2;

    // no 'int' definitions of pow function
	float rowDistance = pow((float)r - maskMidRow, 2);
    float colDistance = pow((float)c - maskMidCol, 2);
    int distance = sqrt(rowDistance + colDistance);

    float value = 1.0 - 0.25 * distance;
    
    // if condition needed for the corners of 7x7 that are calculating
    // to be a little further than I expect them to be
    if (value < 0.25)
        return 0.25;
    else
        return value;
}

__kernel void convolve(
	const __global  uint * const input,
    __constant uint * const mask,
    __global  uint * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    uint sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
            /*
            if (x == 0 && y == 0) {
                printf("(%d, %d) = %d * %d * %f\n", r, c, mask[(r * maskWidth) + c], input[idxIntmp + c], gradientValue(r, c, maskWidth, maskWidth));
            }
            */
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c] * gradientValue(r, c, maskWidth, maskWidth);
        }
    } 
    
	output[y * get_global_size(0) + x] = sum;
}