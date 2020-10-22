all: assignment.cu
	nvcc assignment.cu -std=c++11 -lcudart -lcuda -lcusparse -lcusolver -lcublas -I common/inc -o assignment.exe

fft: cufft_example.cu
	nvcc cufft_example.cu -lcudart -lcuda -lcufft -I common/inc -o cufft_example

blas: cublas_example.cu cusparse_example.cu cusolver_example.cu
	nvcc cublas_example.cu -lcudart -lcuda -lcusparse -lcusolver -lcublas -I common/inc -o cublas_example
	nvcc cusparse_example.cu -lcudart -lcuda -lcusparse -lcusolver -lcublas -I common/inc -o cusparse_example
	nvcc cusolver_example.cu -lcudart -lcuda -lcusparse -lcusolver -lcublas -I common/inc -o cusolver_example

rand: curand_example.cu
	nvcc curand_example.cu -lcudart -lcuda -lcurand -I common/inc -o curand_example

