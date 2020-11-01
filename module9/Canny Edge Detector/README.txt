Build:
make

Run:
# Reads Lena.pgm and outputs an image that shows "edges"
./cannyEdgeDetectorNPP

# Read in a color picture
./cannyEdgeDetectorNPP --color --input person.ppm

./cannyEdgeDetectorNPP --color --input flower.ppm
