Build:

cd assignment
make

Run:
# Defaults
./openCLMath.exe

# Verbose output
./openCLMath.exe --debug

# Change arraysize and/or number of command queues per kernel
./openCLMath.exe --arraysize 10 --queues 2

# Use random values for input (the format must match as shown below with square brackets)
./openCLMath.exe --randomRange [5,10]

# All parameters
./openCLMath.exe --debug --printkernels --arraysize 10 --randomRange [5,10] --queues 2

# Large value tests
./openCLMath.exe --arraysize 1000000 --randomRange [1,1000] --queues 5
