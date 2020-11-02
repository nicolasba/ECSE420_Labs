
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"

//constant vars
#define SIZE 512
#define TOTAL_S (SIZE * SIZE)

//defined in lab2
#define G 0.75 //boundary gain
#define RHO 0.5 
#define ETA 0.0002


//helper funcs (easier to use defines than functions since simple computation of args)
#define index(x, y)		((x) * SIZE + y)
#define row(idx)		(idx / SIZE)
#define column(idx)		(idx % SIZE)

//get/overwrite values with rows and column
#define value(x,y, array) array[(x)*SIZE + y] 
#define overwriteValue(x,y, array, newVal) array[(x)*SIZE + y] = newVal 


#define ROW_CELLS_PER_THREAD 4 // multiply any row per by itself (^2) to get cells/thread, threads/block, etc
#define ROW_THREADS_PER_BLOCK 8
#define ROW_BLOCKS_PER_GRID 16  
#define ROW_CELLS_PER_BLOCK (ROW_CELLS_PER_THREAD * ROW_THREADS_PER_BLOCK)




__global__ void iterate(float* outputVal, float* elements1, float* elements2) {
	int x = ROW_CELLS_PER_BLOCK *blockIdx.x + ROW_CELLS_PER_THREAD*threadIdx.x;
	int y = ROW_CELLS_PER_BLOCK * blockIdx.y + ROW_CELLS_PER_THREAD * threadIdx.y;
	int xEnd = x + ROW_CELLS_PER_THREAD;
	int yEnd = y + ROW_CELLS_PER_THREAD;
	float val = 0;


	//interior
	for (int i = x; i < xEnd; i++) {
		for (int j = y; j < yEnd; j++) {
			if (i > 0 && i < SIZE - 1 && j > 0 && j < SIZE - 1) { 
				val = value(i - 1, j, elements1) + value(i + 1, j, elements1) + value(i, j + 1, elements1) + value(i, j - 1, elements1);
				val -= 4 * value(i, j, elements1);
				val *= RHO;
				val += 2 * value(i, j, elements1) - (1 - ETA) * value(i, j, elements2);
				val /= (1 + ETA);
				overwriteValue(i, j, outputVal, val);
			}

				
		}
	}

	if (x == 0 || xEnd == SIZE || y == 0 || yEnd == SIZE) {
		__syncthreads();
	}

	//Boundary cond
	int temp = 0;
	if (x == 0) { //top
		temp = x;
		for (int i = y; i < yEnd; i++) {
			val = G * value(temp+1, i, outputVal);
			overwriteValue(temp, i, outputVal, val);
		}
	}
	else if (xEnd == SIZE) { //bottom
		temp = xEnd - 1;
		for (int i = y; i < yEnd; i++) {
			val = G * value(temp-1, i, outputVal);
			overwriteValue(temp, i, outputVal, val);
		}
	}
	else if (y == 0) { //left
		temp = y;
		for (int i = x; i < xEnd; i++) {
			val = G * value(i, temp+1, outputVal);
			overwriteValue(i, temp, outputVal, val);
		}
	}
	else if (yEnd == SIZE) { //right
		temp = yEnd - 1;
		for (int i = x; i < xEnd; i++) {
			val = G * value(i, temp-1, outputVal);
			overwriteValue(i, temp, outputVal, val);
		}
	}



	if ((x == 0 && y == 0)||(x==0 && yEnd == SIZE)||(xEnd == SIZE && y == 0)||(xEnd == SIZE && yEnd == SIZE)) {
		__syncthreads();
	}

	//Corners
	if (x == 0 && y == 0) { //topL
		
		val = G * value(1, 0, outputVal);
		overwriteValue(0, 0, outputVal, val);
		
	}
	else if (x == 0 && yEnd == SIZE) { // topR
		
		val = G * value(0, SIZE - 2, outputVal);
		overwriteValue(0, SIZE-1, outputVal, val);
		
	}
	else if (xEnd == SIZE && y == 0) { //|_
		
		val = G * value(SIZE - 2, 0, outputVal);
		overwriteValue(SIZE-1, 0, outputVal, val);
		
	}
	else if (xEnd == SIZE && yEnd == SIZE) { // _|
		
		val = G * value(SIZE - 1, SIZE - 2, outputVal); //CONFIRM THIS
		overwriteValue(SIZE - 1, SIZE - 1, outputVal, val);
		
	}



	if (x == 0 && y == 0) {
		printf("(%d,%d): %f,\n", SIZE / 2, SIZE / 2, value(SIZE/2, SIZE/2, outputVal));
	}
}

int main(int argc, char** argv)
{
	printf("Size of grid: %d nodes\n", TOTAL_S);
	GpuTimer gTim;
	int iterations = atoi(argv[1]);

	float* startValues;
	startValues = (float*)malloc(sizeof(float) * TOTAL_S);
	for (int i = 0; i < TOTAL_S; i++) {
		startValues[i] = 0.0; //init all values at 0
	}

	float* outputVal;
	float* elements1;
	float* elements2;

	//allocated mem
	cudaMalloc(&outputVal, TOTAL_S * sizeof(float));
	cudaMalloc(&elements1, TOTAL_S * sizeof(float));
	cudaMalloc(&elements2, TOTAL_S * sizeof(float));

	//array copy to gpu
	cudaMemcpy(outputVal, startValues, TOTAL_S * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(elements2, startValues, TOTAL_S * sizeof(float), cudaMemcpyHostToDevice);
	overwriteValue(SIZE / 2, SIZE / 2, startValues, 1.0);
	cudaMemcpy(elements1, startValues, TOTAL_S * sizeof(float), cudaMemcpyHostToDevice);

	dim3 Block(ROW_THREADS_PER_BLOCK, ROW_THREADS_PER_BLOCK, 1);
	dim3 Grid(ROW_BLOCKS_PER_GRID, ROW_BLOCKS_PER_GRID, 1);

	double time_used = 0;
	gTim.Start();

	for (int i = 0; i < iterations; i++) {
		// launch the kernel
		iterate << <Grid, Block >> > (outputVal, elements1, elements2);

		float* temp = elements2;
		elements2 = elements1;
		elements1 = outputVal;
		outputVal = temp;
	}
	gTim.Stop();
	time_used = gTim.Elapsed();
	printf("%f", time_used);
	// copy to CPU
	cudaMemcpy(startValues, elements1, TOTAL_S * sizeof(float), cudaMemcpyDeviceToHost);
	free(startValues);


	return 0;
}
