
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

#define BLOCK_QUEUE_CAPACITY 32
//#define BLOCK_QUEUE_CAPACITY 64

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

__global__ void shared_queuing_kernel(GateGraph* gate_graph, int num_threads)
{
#if __CUDA_ARCH__ >= 200
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// Iterate through nodes in the current level
	while (idx < gate_graph->numCurrLevelNodes)
	{
		__shared__ int block_queue[BLOCK_QUEUE_CAPACITY];
		__shared__ int sh_block_queue_idx;
		int node = gate_graph->currLevelNodes_h[idx];
		sh_block_queue_idx = 0;

		// Wait until all threads in block have set block_queue_idx to 0
		__syncthreads();

		// Iterate through neighbours of the current node
		for (int neigh_ptr = gate_graph->nodePtrs_h[node]; neigh_ptr < gate_graph->nodePtrs_h[node + 1]; neigh_ptr++)
		{
			int neigh = gate_graph->nodeNeighbors_h[neigh_ptr];

			// Operate on neighbour if it has not been visited
			if (!atomicCAS(&(gate_graph->nodeVisited_h[neigh]), 0, 1))
			{
				// Update node output
				int gate[3] = { gate_graph->nodeInput_h[neigh], gate_graph->nodeOutput_h[node],
									gate_graph->nodeGate_h[neigh] };
				gate_graph->nodeOutput_h[neigh] = computeGate(gate);

				// Add to block queue if queue not full
				int block_queue_idx = atomicAdd(&(sh_block_queue_idx), 1);				// Increment nextLevelNode index atomically
				if (block_queue_idx < BLOCK_QUEUE_CAPACITY)
				{
					block_queue[block_queue_idx] = neigh;
				}

				// If block queue is full, add to global queue
				else
				{
					int nextLevelNodeIdx = atomicAdd(&(gate_graph->numNextLevelNodes_h), 1);	// Increment nextLevelNode index atomically
					gate_graph->nextLevelNodes_h[nextLevelNodeIdx] = neigh;
				}
			}
			//#endif
		}

		// Wait for all threads in block to write to shared queue
		__syncthreads();

		if (threadIdx.x == 0)
		{
			if (sh_block_queue_idx < BLOCK_QUEUE_CAPACITY)
			{
				int nextLevelNodeIdx = atomicAdd(&(gate_graph->numNextLevelNodes_h), sh_block_queue_idx);
				for (int i = 0; i < sh_block_queue_idx; i++)
				{
					gate_graph->nextLevelNodes_h[nextLevelNodeIdx + i] = block_queue[i];
				}
			}
			else
			{
				int nextLevelNodeIdx = atomicAdd(&(gate_graph->numNextLevelNodes_h), BLOCK_QUEUE_CAPACITY);
				for (int i = 0; i < BLOCK_QUEUE_CAPACITY; i++)
				{
					gate_graph->nextLevelNodes_h[nextLevelNodeIdx + i] = block_queue[i];
				}
			}
		}

		idx += num_threads;

		// Wait for thread 0 to copy content of shared queue to global queue before clearing
		// queue in next iteration
		__syncthreads();
	}
#endif
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
	char* nodeOutput_filename = "nodeOutput_shared.raw";
	char* nextLevelNodes_filename = "nextLvlOutput_shared.raw";

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

	int blockSize[] = { 32, 64 };
	int numBlock[] = { 25, 35 };

	/*for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{*/
	shared_queuing_kernel << < 25, 32 >> > (device_gates, 25 * 32);

	//}

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