Build:
make

Run:
# Reads Lena.pgm and outputs an image that shows "edges"
./cannyEdgeDetectorNPP

# Read in a color picture (currently, not fully functioning. However, the execution times represent what should happen)
# The gray image does not look how I would expect it to look. I guess that my call to nppiRGBToGray_8u_C3C1R is incorrect,
# though it seems like a pretty simple call. While debugging, my output for the color image looks correct, perhaps I am not
# actually loading it properly.
./cannyEdgeDetectorNPP --color --input person.ppm

./cannyEdgeDetectorNPP --color --input flower.ppm
