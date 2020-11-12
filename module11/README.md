Build:
cd Modified\ Convolution
make

Run:
# Default
./assignment.exe

# Verbose
./assignment.exe --debug

# Random data and value control
# Note: you need to look at the "expected computation" to get a look at
# what the some of the input signal and mask look like. Most of the output shows the
# result.
./assignment.exe --debug --random --min 5 --max 20 --maskProbability 0.25

# The mask probability is the probability that a specific cell in a mask is a '1' value.
# Probabilities of --maskProbability 1.0 and --maskProbability 0.0 are interesting parameter values.
