#pragma once
#ifndef SIGNAL_H
#define SIGNAL_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <random>

template<int SIGNALHEIGHT, int SIGNALWIDTH, int MASKHEIGHT, int MASKWIDTH>
struct Signal {
public:
	void populateData() {
		for (int r = 0; r < inputSignalHeight; r++) {
			for (int c = 0; c < inputSignalWidth; c++) {
				inputSignal[r][c] = r * inputSignalWidth + c;
			}
		}

		for (int r = 0; r < maskHeight; r++) {
			for (int c = 0; c < maskWidth; c++) {
				mask[r][c] = 1;
			}
		}
	}

	// maskProbability: the chance that a particular mask cell will be set. Range: [0.00, 1.00]
	void populateRandomData(int min, int max, float maskProbability) {
		srand(time(NULL));

		const int range = max - min;

		for (int r = 0; r < inputSignalHeight; r++) {
			for (int c = 0; c < inputSignalWidth; c++) {
				inputSignal[r][c] = (rand() % range) + min;
			}
		}

		const int percentAsInt = maskProbability * 100;
		for (int r = 0; r < maskHeight; r++) {
			for (int c = 0; c < maskWidth; c++) {
				if ((rand() % 100) < percentAsInt) {
					mask[r][c] = 1;
				}
				else {
					mask[r][c] = 0;
				}
			}
		}
	}

	const unsigned int inputSignalHeight = SIGNALHEIGHT;
	const unsigned int inputSignalWidth = SIGNALWIDTH;

	cl_uint inputSignal[SIGNALHEIGHT][SIGNALWIDTH];

	const unsigned int maskHeight = MASKHEIGHT;
	const unsigned int maskWidth = MASKWIDTH;

	cl_uint mask[MASKHEIGHT][MASKWIDTH];

	static_assert((MASKHEIGHT & 1) == 1, "Mask height is not odd");
	const unsigned int outputSignalHeight = SIGNALHEIGHT - (MASKHEIGHT - 1);

	static_assert((MASKWIDTH & 1) == 1, "Mask width is not odd");
	const unsigned int outputSignalWidth = SIGNALWIDTH - (MASKWIDTH - 1);

	cl_uint outputSignal[SIGNALHEIGHT - (MASKHEIGHT - 1)][SIGNALWIDTH - (MASKWIDTH - 1)];
};


#endif // !SIGNAL_H

