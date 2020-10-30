Build:

make

Example run:
./nvGraph_assignment

./nvGraph_assignment --nodes 4 --verbose
./nvGraph_assignment --nodes 4 --connected --verbose

./nvGraph_assignment --nodes 20
./nvGraph_assignment --nodes 20 --connected

# Large tests
# Many nodes
./nvGraph_assignment --nodes 10000000

# Many edges (any higher than this will result in a memory dump on vocareum)
./nvGraph_assignment --nodes 10000 --connected
