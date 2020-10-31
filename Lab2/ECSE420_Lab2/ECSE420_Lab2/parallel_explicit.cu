
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>

#define AND     0
#define OR      1
#define NAND    2
#define NOR     3
#define XOR     4
#define XNOR    5

#define THREADS_PER_BLOCK 1024

void compareFiles(char* file_name1, char* file_name2)
{
	//get from https://www.tutorialspoint.com/c-program-to-compare-two-files-and-report-mismatches
	FILE* fp1 = fopen(file_name1, "r");
	FILE* fp2 = fopen(file_name2, "r");
	// fetching character of two file 
	// in two variable ch1 and ch2 
	char ch1 = getc(fp1);
	char ch2 = getc(fp2);

	// error keeps track of number of errors 
	// pos keeps track of position of errors 
	// line keeps track of error line 
	int error = 0, pos = 0, line = 1;

	// iterate loop till end of file 
	while (ch1 != EOF && ch2 != EOF)
	{
		pos++;

		// if both variable encounters new 
		// line then line variable is incremented 
		// and pos variable is set to 0 
		if (ch1 == '\n' && ch2 == '\n')
		{
			line++;
			pos = 0;
		}

		// if fetched data is not equal then 
		// error is incremented 
		if (ch1 != ch2)
		{
			error++;
			printf("Line Number : %d \tError"
				" Position : %d \n", line, pos);
		}

		// fetching character until end of file 
		ch1 = getc(fp1);
		ch2 = getc(fp2);
	}

	printf("Total Errors : %d\t", error);
}

// Returns max value from array (only callable by device, host cannot call)
__device__ char computeGate(char values[])
{
	char res = 0;
	char val1 = values[0];
	char val2 = values[1];
	char logic = values[2];


	if (logic == AND) {
		res = val1 & val2;
	}
	else if (logic == OR) {
		res = val1 | val2;

	}
	else if (logic == NAND) {
		res = !(val1 & val2);

	}
	else if (logic == NOR) {
		res = !(val1 | val2);

	}
	else if (logic == XOR) {
		res = val1 ^ val2;

	}
	else if (logic == XNOR) {
		res = !(val1 ^ val2);
	}

	return res + '0';
}

// Kernel function
__global__ void gateKernel(char* d_input_gates, char* d_output_gates, size_t max_size)
{
	// Index for output array is simply the thread index. The index for the input array has to take into consideration
	// the fact that there are 3 values per line
	size_t out_i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t in_i = out_i * 3 * sizeof(char);

	if (out_i < max_size)
	{
		char values[] = { d_input_gates[in_i], d_input_gates[in_i + 1], d_input_gates[in_i + 2] };
		d_output_gates[out_i] = computeGate(values);
	}
}

int main(int argc, char* argv[])
{
	// Command line inputs
	char* input_filename = argv[1];
	size_t fileLines = atoi(argv[2]);
	char* output_filename = argv[3];

	// Host/Device variables
	size_t input_size, output_size;
	char* host_gates_in, * host_gates_out;
	char* device_gates_in, * devices_gates_out;
	size_t number_blocks;
	float data_copy_time, sim_time, total_copy_time;
	GpuTimer timer{};

	//Files
	FILE* file = fopen(input_filename, "r"); //read
	FILE* outFile = fopen(output_filename, "w"); //write from scratch

	// 3 values per input gate/line, 1 per output gate/line
	input_size = 3 * fileLines * sizeof(char);
	output_size = fileLines * sizeof(char);

	// We need #blocks = #lines / #threads per block,  in order to have as many threads as logic gates
	number_blocks = (fileLines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	// Allocate host memory for input and output values
	host_gates_in = (char*)malloc(input_size);
	host_gates_out = (char*)malloc(output_size);

	// Allocate device memory for input and output values
	cudaMalloc((void**)&device_gates_in, input_size);
	cudaMalloc((void**)&devices_gates_out, output_size);

	// Store values from input text file in host memory
	size_t i = 0;
	char line[7];
	while (fgets(line, sizeof(line), file)) {
		char* copy = strdup(line);
		char ops[3] = { 0,0,0 };
		char* token = strtok(copy, ",");
		ops[0] = (char)atoi(token);
		token = strtok(NULL, ",");
		ops[1] = (char)atoi(token);
		token = strtok(NULL, ",");
		ops[2] = (char)atoi(token);

		memcpy(host_gates_in + i * 3 * sizeof(char), ops, 3 * sizeof(char));
		i++;
	}

	printf("Running logic gate simulation in parallel (explicit memory allocation) on \"%s\".\n", input_filename);

	//Copy input data from host to device
	timer.Start();
	cudaMemcpy(device_gates_in, host_gates_in, input_size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	timer.Stop();
	total_copy_time = data_copy_time = timer.Elapsed();
	printf("Copying input data from host mem to device mem took %f ms (or %f s).\n", data_copy_time,
		data_copy_time / 1000);


	sim_time = 0;
	/*for (i = 0; i < 10; i++)
	{*/
	// Launch kernel
	printf("There are %d threads running.\n", fileLines);
	timer.Start();
	gateKernel << <number_blocks, THREADS_PER_BLOCK >> > (device_gates_in, devices_gates_out, fileLines);
	cudaDeviceSynchronize();
	timer.Stop();
	sim_time += timer.Elapsed();
	//}
	//printf("Execution time: %f ms (or %f s).\n", sim_time / 10, sim_time / 10 / 1000);
	printf("Execution time: %f ms (or %f s).\n", sim_time, sim_time / 1000);


	// Copy output data from device to host
	timer.Start();
	cudaMemcpy(host_gates_out, devices_gates_out, output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	timer.Stop();
	data_copy_time = timer.Elapsed();
	total_copy_time += data_copy_time;
	printf("Copying output data from device mem to host mem took %f ms (or %f s).\n", data_copy_time,
		data_copy_time / 1000);

	printf("Total data migration time: %f ms (or %f s), execution time: %f ms (or %f s).\n", total_copy_time, total_copy_time / 1000,
		sim_time, sim_time / 1000);

	// Output to file
	for (i = 0; i < fileLines; i++)
	{
		fprintf(outFile, "%c\n", host_gates_out[i]);
	}
	printf("The output has been saved to \"%s\".\n", output_filename);

	// Cleanup
	free(host_gates_in); free(host_gates_out);
	cudaFree(device_gates_in); cudaFree(devices_gates_out);
	fclose(outFile); fclose(file);

	//compareFiles("out_exp_1000000.txt", "sol_1000000.txt");
	printf("\n");
	return 0;
}

