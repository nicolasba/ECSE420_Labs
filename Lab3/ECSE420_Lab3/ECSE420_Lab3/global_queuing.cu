
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "read_input.h"
#include "write_output.h"
#include "gputimer.h"

#include <stdio.h>
#include <stdlib.h>

#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5

typedef struct GateGraph {
	// Input vars
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

	// Output vars
	int numNextLevelNodes_h; // Queue
	int* nextLevelNodes_h;
} GateGraph;

__device__ int computeGate(int values[]) {
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
}

__global__ void global_queuing_kernel(GateGraph* gate_graph, int num_threads)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Iterate through nodes in the current level
	while (idx < gate_graph->numCurrLevelNodes)
	{
		int node = gate_graph->currLevelNodes_h[idx];

		// Iterate through neighbours of the current node
		for (int neigh_ptr = gate_graph->nodePtrs_h[node]; neigh_ptr < gate_graph->nodePtrs_h[node + 1]; neigh_ptr++)
		{
			int neigh = gate_graph->nodeNeighbors_h[neigh_ptr];

#if __CUDA_ARCH__ >= 200
			// Operate on neighbour if it has not been visited
			if (!atomicCAS(&(gate_graph->nodeVisited_h[neigh]), 0, 1))
			{
				int gate[3] = { gate_graph->nodeInput_h[neigh], gate_graph->nodeOutput_h[node],
									gate_graph->nodeGate_h[neigh] };

				int nextLevelNodeIdx = atomicAdd(&(gate_graph->numNextLevelNodes_h), 1);	// Increment nextLevelNode index atomically
				gate_graph->nodeOutput_h[neigh] = computeGate(gate);
				gate_graph->nextLevelNodes_h[nextLevelNodeIdx] = neigh;
			}
#endif
		}

		idx += num_threads;
	}
}

int main(int argc, char* argv[])
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
	char* nodeOutput_filename = "nodeOutput_global.raw";
	char* nextLevelNodes_filename = "nextLvlOutput_global.raw";

	// Timer object
	GpuTimer timer{};

	// Graph structs (hostmem / devicemem)
	GateGraph* host_gates;
	GateGraph* temp_device_gates, * device_gates;

	// Allocate host / device memory
	host_gates = (GateGraph*)malloc(sizeof(GateGraph));
	temp_device_gates = (GateGraph*)malloc(sizeof(GateGraph));
	cudaMalloc((void**)&device_gates, sizeof(GateGraph));

	// Parse input
	host_gates->numNodePtrs = read_input_one_two_four(&(host_gates->nodePtrs_h), input1_filename);
	host_gates->numTotalNeighbors_h = read_input_one_two_four(&(host_gates->nodeNeighbors_h), input2_filename);
	host_gates->numNodes = read_input_three(&(host_gates->nodeVisited_h), &(host_gates->nodeGate_h),
		&(host_gates->nodeInput_h), &(host_gates->nodeOutput_h), input3_filename);
	host_gates->numCurrLevelNodes = read_input_one_two_four(&(host_gates->currLevelNodes_h), input4_filename);
	// Allocate mem for nextLvlNodes (same size as number of nodes - will later count how many nodes there actually are)
	host_gates->nextLevelNodes_h = (int*)malloc(host_gates->numNodes * sizeof(int));
	host_gates->numNextLevelNodes_h = 0;

	// Set array sizes in device memory
	temp_device_gates->numNodePtrs = host_gates->numNodePtrs;
	temp_device_gates->numTotalNeighbors_h = host_gates->numTotalNeighbors_h;
	temp_device_gates->numNodes = host_gates->numNodes;
	temp_device_gates->numCurrLevelNodes = host_gates->numCurrLevelNodes;
	temp_device_gates->numNextLevelNodes_h = host_gates->numNextLevelNodes_h;

	// Allocate space for arrays in device memory
	cudaMalloc((void**)&(temp_device_gates->nodePtrs_h), temp_device_gates->numNodePtrs * sizeof(int));
	cudaMalloc((void**)&(temp_device_gates->nodeNeighbors_h), temp_device_gates->numTotalNeighbors_h * sizeof(int));
	cudaMalloc((void**)&(temp_device_gates->nodeVisited_h), temp_device_gates->numNodes * sizeof(int));
	cudaMalloc((void**)&(temp_device_gates->nodeGate_h), temp_device_gates->numNodes * sizeof(int));
	cudaMalloc((void**)&(temp_device_gates->nodeInput_h), temp_device_gates->numNodes * sizeof(int));
	cudaMalloc((void**)&(temp_device_gates->nodeOutput_h), temp_device_gates->numNodes * sizeof(int));
	cudaMalloc((void**)&(temp_device_gates->currLevelNodes_h), temp_device_gates->numCurrLevelNodes * sizeof(int));
	//Allocate global queue (we do not copy into it)
	cudaMalloc((void**)&(temp_device_gates->nextLevelNodes_h), temp_device_gates->numNodes * sizeof(int));

	// Copy array contents to device memory
	cudaMemcpy(temp_device_gates->nodePtrs_h, host_gates->nodePtrs_h, temp_device_gates->numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_device_gates->nodeNeighbors_h, host_gates->nodeNeighbors_h, temp_device_gates->numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_device_gates->nodeVisited_h, host_gates->nodeVisited_h, temp_device_gates->numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_device_gates->nodeGate_h, host_gates->nodeGate_h, temp_device_gates->numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_device_gates->nodeInput_h, host_gates->nodeInput_h, temp_device_gates->numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_device_gates->nodeOutput_h, host_gates->nodeOutput_h, temp_device_gates->numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_device_gates->currLevelNodes_h, host_gates->currLevelNodes_h, temp_device_gates->numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);

	// Copy struct from host to device
	cudaMemcpy(device_gates, temp_device_gates, sizeof(GateGraph), cudaMemcpyHostToDevice);

	int blockSize[] = { 32, 64, 128 };
	int numBlock[] = { 10, 25, 35 };

	// Invoke kernel on different blk_size and num_blks configurations
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			// Invoke kernel and compute elapsed time
			timer.Start();

			global_queuing_kernel << <numBlock[i], blockSize[j] >> > (device_gates, numBlock[i] * blockSize[j]);
			cudaDeviceSynchronize();

			timer.Stop();
			printf("%d %d %f\n", numBlock[i], blockSize[j], timer.Elapsed());
		}

	// Copy struct from device to device
	cudaMemcpy(temp_device_gates, device_gates, sizeof(GateGraph), cudaMemcpyDeviceToHost);

	// Copy nodeOutput and nextLevelNodes from device to host
	cudaMemcpy(host_gates->nodeOutput_h, temp_device_gates->nodeOutput_h, temp_device_gates->numNodes * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_gates->nextLevelNodes_h, temp_device_gates->nextLevelNodes_h, temp_device_gates->numNextLevelNodes_h * sizeof(int), cudaMemcpyDeviceToHost);
	host_gates->numNextLevelNodes_h = temp_device_gates->numNextLevelNodes_h;

	// Write nodeOutput and nextLevelNodes to files
	write_output(host_gates->nodeOutput_h, host_gates->numNodes, nodeOutput_filename);
	write_output(host_gates->nextLevelNodes_h, host_gates->numNextLevelNodes_h, nextLevelNodes_filename);

	return 0;
}