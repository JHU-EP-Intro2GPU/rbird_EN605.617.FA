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
# Add the debug flag to view the values
./openCLMath.exe --randomRange [5,10] --debug

# All parameters
./openCLMath.exe --debug --printkernels --arraysize 10 --randomRange [5,10] --queues 2

# Large value tests
./openCLMath.exe --arraysize 1000000 --randomRange [1,1000] --queues 5

# Tests that compare runtime based on the number of queues (1,000,000 elements and change the number of queues)
# These same tests were performed with 10,000,000 numbers as well and shows more significant
# improvement by adding more queues.
ccc_v1_w_DN6k_191794@runweb6:~/module13/assignment$ ./openCLMath.exe --arraysize 1000000 --queues 1
ccc_v1_w_DN6k_191794@runweb6:~/module13/assignment$ ./openCLMath.exe --arraysize 1000000 --queues 2
ccc_v1_w_DN6k_191794@runweb6:~/module13/assignment$ ./openCLMath.exe --arraysize 1000000 --queues 4
ccc_v1_w_DN6k_191794@runweb6:~/module13/assignment$ ./openCLMath.exe --arraysize 1000000 --queues 10
