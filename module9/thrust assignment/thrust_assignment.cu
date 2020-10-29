
#include "assignment.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <random>
#include <time.h>

#include <iostream>

// print a host vector in array format: [1,2,3,4]
template<typename T>
std::ostream& operator<<(std::ostream& out, const thrust::host_vector<T>& vect) {
    out << "[";
    for (size_t i = 0; i < vect.size(); i++) {
        if (i != 0) {
            out << ",";
        }
        out << vect[i];
    }
    out << "]";
    return out;
}

struct CommandLineArgs {
public:
    CommandLineArgs(int argc, const char* argv[]) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--elements") == 0) {
                elements = atoi(argv[++i]);
            }
            else if (strcmp(arg, "--random") == 0) {
                randomElements = true;
            }
            else if (strcmp(arg, "--debug") == 0) {
                debug = true;
            }
        }
    }

    int elements = 32;
    bool randomElements = false;
    bool debug = false;
};

constexpr int RandomMin = 1;
constexpr int RandomMax = 100;

inline int randomValue() {
    return RandomMin + (rand() % RandomMax);
}

template <typename T>
void testThrustOperators(const CommandLineArgs& args, std::string typeName) {
    // Device Vector subscript operator ([]) does a memcpy on EACH call. Avoid its use
    thrust::host_vector<T> inputA(args.elements), inputB(args.elements);

    if (args.randomElements) {
        for (size_t i = 0; i < args.elements; i++) {
            inputA[i] = randomValue();
            inputB[i] = randomValue();
        }
    }
    else {
        thrust::sequence(inputA.begin(), inputA.end(), 17); // 17 -> elements + 17 (make the modulus operator interesting)
        thrust::sequence(inputB.begin(), inputB.end()); // 0 -> elements
    }

    if (args.debug) {
        std::cout << "InputA:" << std::endl << inputA << std::endl;
        std::cout << "InputB:" << std::endl << inputB << std::endl;
    }

    thrust::device_vector<T> d_inputA(inputA), d_inputB(inputB);

    // Only have output vectors on device. No need to verify correct values from thrust (except for debug)
    thrust::device_vector<T> d_outputAdd(args.elements), d_outputSub(args.elements);
    thrust::device_vector<T> d_outputMult(args.elements), d_outputMod(args.elements);

    std::cout << std::endl;

    // Make transform calls and report times
    {
        TimeCodeBlockCuda timeAdd(typeName + " Add");
        thrust::transform(d_inputA.begin(), d_inputA.end(), d_inputB.begin(), d_outputAdd.begin(), thrust::plus<T>());
    }
    {
        TimeCodeBlockCuda timeAdd(typeName + " Subtract");
        thrust::transform(d_inputA.begin(), d_inputA.end(), d_inputB.begin(), d_outputSub.begin(), thrust::minus<T>());
    }
    {
        TimeCodeBlockCuda timeAdd(typeName + " Multiply");
        thrust::transform(d_inputA.begin(), d_inputA.end(), d_inputB.begin(), d_outputMult.begin(), thrust::multiplies<T>());
    }
    {
        TimeCodeBlockCuda timeAdd(typeName + " Divide");
        thrust::transform(d_inputA.begin(), d_inputA.end(), d_inputB.begin(), d_outputMod.begin(), thrust::modulus<T>());
    }

    if (args.debug) {
        thrust::host_vector<T> outputAdd(d_outputAdd), outputSub(d_outputSub), outputMult(d_outputMult), outputMod(d_outputMod);

        std::cout << "OutputAdd:" << std::endl << outputAdd << std::endl;
        std::cout << "OutputSub:" << std::endl << outputSub << std::endl;
        std::cout << "OutputMult:" << std::endl << outputMult << std::endl;
        std::cout << "OutputMod:" << std::endl << outputMod << std::endl;
    }
}

int main(int argc, const char* argv[])
{
    CommandLineArgs args(argc, argv);

    const char* elementType = (args.randomElements) ? "Random" : "Sequential";
    std::printf("%s Elements: %d\n", elementType, args.elements);

    if (args.randomElements)
        srand(time(NULL));

    // Modulus operator is only compatible with integral values (floats and doubles not supported)
    testThrustOperators<short>(args, "short");
    testThrustOperators<int>(args, "int");
    testThrustOperators<long>(args, "long");

    return 0;
}

