all: assignment.c
	gcc assignment.c -o assignment.exe

simple:
	nvcc simple_asynch.cpp -lOpenCL -o simple_asynch

filter:
	nvcc ImageFilter2D.cpp common/FreeImage/lib/linux/x86_64/libfreeimage.a -lOpenCL -I common/FreeImage/include -o ImageFilter2D

