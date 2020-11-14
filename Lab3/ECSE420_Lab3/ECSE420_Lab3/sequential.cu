
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "read_input.h"
#include "write_output.h"

#include <stdio.h>
#include <stdlib.h>

#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5

int computeOutput(int values[]) {
    int compute;
    int val1 = values[0];
    int val2 = values[1];
    int logic = values[2];


    if (logic == AND) {
        compute = val1 & val2;
    }
    else if (logic == OR) {
        compute = val1 | val2;

    }
    else if (logic == NAND) {
        compute = !(val1 & val2);

    }
    else if (logic == NOR) {
        compute = !(val1 | val2);

    }
    else if (logic == XOR) {
        compute = val1 ^ val2;

    }
    else if (logic == XNOR) {
        compute = !(val1 ^ val2);
    }

    return compute;
    /*char toReturn = compute + '0';
    return toReturn;*/
}

int main(int argc, char *argv[])
{
    // Command line
    /*char* input1_filename = argv[1];
    char* input2_filename = argv[2];
    char* input3_filename = argv[3];
    char* input4_filename = argv[4];
    char* nodeOutput_filename = argv[5];
    char* nextLevelNodes_filename = argv[6];*/

    char* input1_filename = "input1.raw";
    char* input2_filename = "input2.raw";
    char* input3_filename = "input3.raw";
    char* input4_filename = "input4.raw";
    char* nodeOutput_filename = "nodeOutput_seq.raw";
    char* nextLevelNodes_filename = "nextLvlOutput_seq.raw";

    // Input variables
    int numNodePtrs;
    int* nodePtrs_h;
    int numTotalNeighbors_h;
    int* nodeNeighbors_h;
    int numNodes;
    int* nodeVisited_h;
    int* nodeGate_h;
    int* nodeInput_h;
    int* nodeOutput_h;
    int numCurrLevelNodes;
    int* currLevelNodes_h;

    // Output
    int numNextLevelNodes_h;
    int* nextLevelNodes_h;

    // Parse input
    numNodePtrs = read_input_one_two_four(&nodePtrs_h, input1_filename);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input2_filename);
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input3_filename);
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input4_filename);

    // Allocate mem for nextLvlNodes
    nextLevelNodes_h = (int*)malloc(numNodes * sizeof(int));
    numNextLevelNodes_h = 0;

    // Iterate through nodes in the current level
    for (int i = 0; i < numCurrLevelNodes; i++)
    {
        int node = currLevelNodes_h[i];

        // Iterate through neighbours of the current node
        for (int neigh_ptr = nodePtrs_h[node]; neigh_ptr < nodePtrs_h[node + 1]; neigh_ptr++)
        {
            int neigh = nodeNeighbors_h[neigh_ptr];
            
            if (!nodeVisited_h[neigh])
            {
                nodeVisited_h[neigh] = 1;

                int gate[3] = {nodeInput_h[neigh], nodeOutput_h[node], nodeGate_h[neigh]};

                nodeOutput_h[neigh] = computeOutput(gate);
                nextLevelNodes_h[numNextLevelNodes_h++] = neigh;
            }
        }
    }

    write_output(nodeOutput_h, numNodes, nodeOutput_filename);
    write_output(nextLevelNodes_h, numNextLevelNodes_h, nextLevelNodes_filename);

    return 0;
}