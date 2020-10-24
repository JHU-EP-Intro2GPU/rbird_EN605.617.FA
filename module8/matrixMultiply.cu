
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "assignment.h"
#include "kernels.h"

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <string.h>

#define INDEX(row,col,cols_per_row) (row * cols_per_row + col)

static const char* _cudaGetErrorEnum(cublasStatus_t error);

#define cublasCheck(stmt) { cublasAssert((stmt), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", _cudaGetErrorEnum(code), file, line);
        if (abort) exit(code);
    }
}

// Perform matrix multiply on two matricies of dimentions:
// (AxB) * (BxC)
struct CommandLineArgs {
public:
    CommandLineArgs(int argc, const char* argv[]) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--A") == 0) {
                A = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--B") == 0) {
                B = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--C") == 0) {
                C = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--random") == 0) {
                randomSeed = true;
            }
            else if (strcmp(arg, "--debug") == 0) {
                debug = true;
            }
        }
    }

    int A = 5;
    int B = 3;
    int C = 4;

    /*
    ///* class example
    int A = 2;
    int B = 9;
    int C = 2;
    */

    bool randomSeed = false;
    bool debug = false;
};


#define VAL_MIN (-50)
#define VAL_MAX (50)

void writeRandomValueTo(float& value) {
    float f = (float)rand() / RAND_MAX;
    value = VAL_MIN + f * (VAL_MAX - VAL_MIN);
}

void writeRandomValueTo(double& value) {
    double f = (double)rand() / RAND_MAX;
    value = VAL_MIN + f * (VAL_MAX - VAL_MIN);
}


template <typename T>
class Matrix : public HostAndDeviceMemory<T>
{
    int cols, rows;

public:
    Matrix(int _rows, int _cols) : HostAndDeviceMemory(_cols * _rows), cols(_cols), rows(_rows) {
    }

    ~Matrix()
    {
    }

    int Rows() const { return rows; }
    int Cols() const { return cols; }


    void print() const {
        std::printf("Matrix[%d x %d]:\n", rows, cols);

        // print in column major order
        for (int c = 0; c < rows; c++) {
            for (int r = 0; r < cols; r++) {
                std::cout << " " << this->host()[INDEX(r, c, rows)];
            }

            std::cout << std::endl;
        }

    }

    void populateData() {
        for (size_t count = 0; count < size(); count++) {
            host()[count] = count; //+ 1;
        }
    }

    void populateRandomData() {
        for (size_t count = 0; count < size(); count++) {
            writeRandomValueTo(host()[count]);
        }
    }
};

// helper functions to call the correct matrix multiply based on type
void performMatrixMultiply(Matrix<float>& matrix1, Matrix<float>& matrix2, Matrix<float>& output, cublasHandle_t handle)
{
    cublasOperation_t nonTranspose = cublasOperation_t::CUBLAS_OP_N; // NonTranspose, 'N'
    int m = matrix1.Rows(); // op ( A ) m × k , op ( B ) k × n and C m × n , respectively. (NVidia documentation
    int n = matrix2.Cols();
    int k = matrix1.Cols();
    float alpha = 1.0; //  scalar used for multiplication
    float beta = 0.0; //  scalar used for multiplication
    int leadingDimensionA = matrix1.Rows(); //m; // matrix1.Rows();
    int leadingDimensionB = matrix2.Rows(); //k; // matrix2.Rows();
    int leadingDimensionC = output.Rows(); //m; // output.Rows();

    // TODO: Test, verify correct answers, and write function for double
    cublasCheck(cublasSgemm(
        handle, nonTranspose, nonTranspose,
        m, n, k,
        &alpha, matrix1.device(), leadingDimensionA,
        matrix2.device(), leadingDimensionB, &beta,
        output.device(), leadingDimensionC));

}

void performMatrixMultiply(Matrix<double>& matrix1, Matrix<double>& matrix2, Matrix<double>& output, cublasHandle_t handle)
{
    // TODO
}

template <typename T>
void testMatrixMultiply(const CommandLineArgs& testArgs, cublasHandle_t handle)
{
    Matrix<T> matrix1(testArgs.A, testArgs.B);
    Matrix<T> matrix2(testArgs.B, testArgs.C);
    Matrix<T> resultMatrix(testArgs.A, testArgs.C);

    if (testArgs.randomSeed) {
        matrix1.populateRandomData();
        matrix2.populateRandomData();
    }
    else {
        matrix1.populateData();
        matrix2.populateData();
    }

    if (testArgs.debug) {
        matrix1.print();    std::cout << std::endl;
        matrix2.print();    std::cout << std::endl;
    }

    cublasCheck(cublasSetMatrix(matrix1.Rows(), matrix1.Cols(), sizeof(T), matrix1.host(), matrix1.Rows(), matrix1.device(), matrix1.Rows()));
    cublasCheck(cublasSetMatrix(matrix2.Rows(), matrix2.Cols(), sizeof(T), matrix2.host(), matrix2.Rows(), matrix2.device(), matrix2.Rows()));

    performMatrixMultiply(matrix1, matrix2, resultMatrix, handle);

    cublasCheck(cublasGetMatrix(resultMatrix.Rows(), resultMatrix.Cols(), sizeof(T), resultMatrix.device(), resultMatrix.Rows(), resultMatrix.host(), resultMatrix.Rows()));

    if (testArgs.debug) {
        std::cout << "Result:" << std::endl;
        resultMatrix.print();   std::cout << std::endl;
    }

}


int main(int argc, const char* argv[])
{
    CommandLineArgs testArgs(argc, argv);
    srand(time(NULL));

    cublasHandle_t cublasContext;
    cublasCheck(cublasCreate(&cublasContext));

    testMatrixMultiply<float>(testArgs, cublasContext);
    //testMatrixMultiply<double>(testArgs, cublasContext);

    cublasCheck(cublasDestroy(cublasContext));

    // cublas v2 does not need to call shutdown (based on Nvidia example)


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    gpuErrchk(cudaDeviceReset());

    return 0;
}


const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
