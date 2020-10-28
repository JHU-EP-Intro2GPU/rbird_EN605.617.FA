
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <random>

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

void writeRandomValue(int& value) {
    value = RandomMin + (rand() % RandomMax);
}

void writeRandomValue(double& value) {
    double f = (double)rand() / RAND_MAX;
    value = RandomMin + f * (RandomMax - RandomMin);
}

// I couldn't get thrust random to compile on windows:
// create a minstd_rand object to act as our source of randomness
/*
thrust::minstd_rand rng;
thrust::random::normal_distribution<T> dist(2, 100);

// won't compile on windows
inputA[i] = dist(rng);
inputB[i] = dist(rng);
*/
template<typename T>
void testThrustOperators(const CommandLineArgs& args) {
    // Device Vector subscript operator ([]) does a memcpy on EACH call. Avoid its use
    thrust::host_vector<T> inputA(args.elements), inputB(args.elements);

    if (args.randomElements) {
        for (size_t i = 0; i < args.elements; i++) {
            writeRandomValue(inputA[i]);
            writeRandomValue(inputB[i]);
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
    thrust::device_vector<T> d_outputAdd, d_outputSub, d_outputMult, d_outputMod;

    // Make transform calls
    thrust::transform(d_inputA.begin(), d_inputA.end(), d_outputAdd, thrust::plus<T>());
    thrust::transform(d_inputA.begin(), d_inputA.end(), d_outputSub, thrust::minus<T>());
    thrust::transform(d_inputA.begin(), d_inputA.end(), d_outputMult, thrust::multiplies<T>());
    thrust::transform(d_inputA.begin(), d_inputA.end(), d_outputMod, thrust::modulus<T>());

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

    testThrustOperators<int>(args);

    return 0;
}

