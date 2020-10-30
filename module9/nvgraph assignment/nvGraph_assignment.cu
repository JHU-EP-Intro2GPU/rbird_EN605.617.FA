
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "nvgraph.h"

// starting point:
// Nvidia example, https://docs.nvidia.com/cuda/nvgraph/index.html#nvgraph-sssp-example

#include <memory>
#include <stdio.h>
#include <vector>

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

struct GraphData
{
    // sample data converted to stl containers
    std::vector<float> weights_h = { 0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5 };
    std::vector<int> destination_offsets_h = { 0, 1, 3, 4, 6, 8, 10 };
    std::vector<int> source_indices_h = { 2, 0, 2, 0, 4, 5, 2, 3, 3, 4 };
    
    size_t numNodes() const { return destination_offsets_h.size() - 1; } // the final index is the number of edges
    size_t numEdges() const { return source_indices_h.size(); }
};

int main()
{    
    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    
    GraphData graphData;
    
    // Init host data
    std::vector<float> sssp_1_h(graphData.numNodes());
    
    std::vector<void*> vertex_dim = { (void*)sssp_1_h.data() };
    std::vector<cudaDataType_t> vertex_dimT = { CUDA_R_32F };
    
    nvgraphCSCTopology32I_t CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    
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
    
    // Solve
    int source_vert = 0;
    check(nvgraphSssp(handle, graph, 0, &source_vert, 0));
    
    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h.data(), 0));
    
    // print shortest distances
    for (int i = 0; i < sssp_1_h.size(); i++)
    {
        std::printf("%d: %f\n", i, sssp_1_h[i]);
    }
    
    //Clean 
    free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    
    return 0;
}

