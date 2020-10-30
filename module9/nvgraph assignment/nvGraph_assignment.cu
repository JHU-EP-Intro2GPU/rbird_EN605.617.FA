
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "nvgraph.h"
#include "assignment.h"

// starting point:
// Nvidia example, https://docs.nvidia.com/cuda/nvgraph/index.html#nvgraph-sssp-example

#include <memory>
#include <stdio.h>
#include <vector>
    
struct CommandLineArgs {
public:
    CommandLineArgs(int argc, const char* argv[]) {
        for (int i = 1; i < argc; i++) {
            const char* arg = argv[i];
            if (strcmp(arg, "--nodes") == 0) {
                nodes = atoi(argv[++i]);
                useSampleData = false;
            }
            else if (strcmp(arg, "--connected") == 0) {
                fullyConnectedGraph = true;
            }
            else if (strcmp(arg, "--verbose") == 0) {
                verbose = true;
            }
        }
    }

    bool useSampleData = true;
    int nodes = 0;
    bool fullyConnectedGraph = false;
    
    bool verbose = false;
};

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

/** Sample graph visualization (as best I can with text)

                 0
              //   \
              2 --> 1
              \
               4 == 3
                \  //
                  5

Run the program to see the exact weights. I have verified the correctness
Of the library's outputs with the current data and modified data.
*/
struct GraphData
{
    // sample data converted to stl containers
    std::vector<float> weights_h = { 0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5 };
    std::vector<int> destination_offsets_h = { 0, 1, 3, 4, 6, 8, 10 };
    std::vector<int> source_indices_h = { 2, 0, 2, 0, 4, 5, 2, 3, 3, 4 };
    
    size_t numNodes() const { return destination_offsets_h.size() - 1; } // the final index is the number of edges
    size_t numEdges() const { return source_indices_h.size(); }
    
    void generateGraph(const CommandLineArgs& args) {
    }
    
    void printEdges() {
        std::printf("Edges:\n");
        int destinationNode = 0;
        for (int i = 0; i < numEdges(); i++) {
            if (i == destination_offsets_h[destinationNode + 1]) {
                destinationNode++;
            }
            std::printf("%d -> %d (%f)\n", source_indices_h[i], destinationNode, weights_h[i]);
        }
    }
};

int main(int argc, const char* argv[])
{
    CommandLineArgs args(argc, argv);
    
    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    
    GraphData graphData;
    std::cout << "Nodes: " << graphData.numNodes() << " Edges: " << graphData.numEdges() << std::endl;

    if (args.verbose)
        graphData.printEdges();
    
    // Init host data
    std::vector<float> sssp_1_h(graphData.numNodes());
    
    std::vector<void*> vertex_dim = { (void*)sssp_1_h.data() };
    std::vector<cudaDataType_t> vertex_dimT = { CUDA_R_32F };
    
    nvgraphCSCTopology32I_t CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    {
        TimeCodeBlockCuda dataCreationAndTransfer("\nGraph initialization and data transfer");
        check(nvgraphCreate(&handle));
        check(nvgraphCreateGraphDescr(handle, &graph));
        CSC_input->nvertices = graphData.numNodes();
        CSC_input->nedges = graphData.numEdges();
        CSC_input->destination_offsets = graphData.destination_offsets_h.data();
        CSC_input->source_indices = graphData.source_indices_h.data();
    
        // Set graph connectivity and properties (tranfers)
        check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
        check(nvgraphAllocateVertexData(handle, graph, vertex_dimT.size(), vertex_dimT.data()));
        check(nvgraphAllocateEdgeData(handle, graph, 1, &edge_dimT));
        check(nvgraphSetEdgeData(handle, graph, (void*)graphData.weights_h.data(), 0));
    }
        
    // Solve
    int source_vert = 0;
    {
        TimeCodeBlockCuda shortestPathsCalculation("Shortest Paths Calculation");
        check(nvgraphSssp(handle, graph, 0, &source_vert, 0));
    }
    
    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h.data(), 0));
    
    if (args.verbose) {
        // print shortest distances
        std::printf("\nShortest distance from 0:\n");
        for (int i = 0; i < sssp_1_h.size(); i++)
        {
            std::printf("%d: %f\n", i, sssp_1_h[i]);
        }
    }
    
    //Clean 
    free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    
    return 0;
}

