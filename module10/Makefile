all: openCLMath.cpp
	nvcc -std=c++11 openCLMath.cpp -lOpenCL -o openCLMath.exe

helloworld: HelloWorld.cpp
	nvcc HelloWorld.cpp -o HelloWorldCL -lOpenCL
