#pragma once
#ifndef TIME_BLOCK_H
#define TIME_BLOCK_H

#include <chrono>

class TimeCodeBlock
{
#ifdef _WIN32
    std::chrono::steady_clock::time_point start;
#else
    std::chrono::system_clock::time_point start;
#endif // _WIN32

    const char* name;
public:
    TimeCodeBlock(const char* blockName) : name(blockName) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~TimeCodeBlock() {
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::microseconds timeDiff = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

        auto microseconds = timeDiff.count() % 1000;
        auto milliseconds = timeDiff.count() / 1000;

        std::cout << name << " Execution time =";
        if (milliseconds != 0) {
            std::cout << " " << milliseconds << " milliseconds";
        }

        if (milliseconds != 0) {
            std::cout << " " << microseconds << " microseconds";
        }

        std::cout << "." << std::endl;
    }
};


#endif // !TIME_BLOCK_H

