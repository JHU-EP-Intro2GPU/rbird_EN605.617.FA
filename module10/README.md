This directory contains some extra files of my first attempt to
refactor the code to be very readable. I tried making some classes
that would set the foundation for future OpenCL assignments. I was
unable to get this to work and instead refocused my efforts on 
making simpler changes to the example code. I hope to have the
helper classes figured out by next assignment (I'm leaving the)
files here just to show that more effort was put into this
assignment than the effort you see in the working file.

Build:
make

Example runs:
./openCLMath.exe

# Run with different buffer sizes (1 million is the highest practical test, pow is slow for my implementation)
# 1 million took 90 seconds to run pow. I could speed this up by limiting the values stored in the A and B input
# buffers
./openCLMath.exe --arraysize 100000

# Output kernel code and result buffers
./openCLMath.exe --arraysize 100000

