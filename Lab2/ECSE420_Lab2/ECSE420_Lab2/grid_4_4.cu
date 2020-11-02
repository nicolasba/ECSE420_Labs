
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


#define PARALLEL 1 //0 sequential, 1 parallel

//constant vars
#define SIZE 4
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

#define idx_overwriteValue(idx, array, newVal) overwriteValue(row(idx),column(idx), array, newVal) 

__global__ void iterate(float* outputVal, float* elements1, float* elements2) {
	int x = threadIdx.x;
	int y = threadIdx.y;

	int idx = index(x, y);
	float val = 0;

	if (idx == 5 || idx == 6 || idx == 9 || idx == 10) { //interior elements
		val = value( x - 1, y, elements1) + value( x + 1, y, elements1) + value( x, y + 1, elements1) + value(x, y - 1, elements1);
		val -= 4 * value(x, y, elements1);
		val *= RHO;
		val += 2 * value( x, y, elements1) - (1 - ETA) * value(x, y, elements2);
		val /= (1 + ETA);
	}
	//overwriteValue(x, y, outputVal, val);
	idx_overwriteValue(idx, outputVal, val);
	__syncthreads();

	//borders
	if (idx == 1 || idx == 2) { // Top
		val = G * value( 1, y, outputVal);
	}
	else if (idx == 4 || idx == 8) { // Left
		val = G * value(x, 1, outputVal);
	}
	else if (idx == 7 || idx == 11) { // Right
		val = G * value( x, SIZE - 2, outputVal);
	}
	else if (idx == 13 || idx == 14) { // Bottom
		val = G * value(SIZE -2 , y, outputVal);
	}
	//overwriteValue(x, y, outputVal, val);

	idx_overwriteValue(idx, outputVal, val);
	__syncthreads();

	if (idx == 0) { //topL
		val = G * value(1, 0, outputVal);
	}
	else if (idx == 3) { //topR
		val = G * value(0, SIZE - 2, outputVal);
	}
	else if (idx == 12) { // |_
		val = G * value(SIZE - 2, 0, outputVal);
	}
	else if (idx == 15) {// _|
		val = G * value(SIZE - 2, SIZE - 2, outputVal);
	}

	idx_overwriteValue(idx, outputVal, val);
	//overwriteValue(x, y, outputVal, val);

	if (x == SIZE/2 && y == SIZE/2) {
		printf("(%d,%d): %f,\n",x,y, value(x, y, outputVal));
	}
}


void iterateSeq(float* outputVal, float* elements1, float* elements2, int iter) {
	//interior
	float val = 0;
	for (int x = 1; x < SIZE - 1; x++) {
		for (int y = 1; y < SIZE - 1; y++) {

			val = value(x - 1, y, elements1) + value(x + 1, y, elements1) + value(x, y + 1, elements1) + value(x, y - 1, elements1);
			val -= 4 * value(x, y, elements1);
			val *= RHO;
			val += 2 * value(x, y, elements1) - (1 - ETA) * value(x, y, elements2);
			val /= (1 + ETA);
			overwriteValue(x, y, outputVal, val);
		}
	}

	//borders
	for (int x = 1; x < SIZE - 1; x++) {
		val = G * value(1, x, outputVal); //T
		overwriteValue(0, x, outputVal, val);
		val = G * value(x, 1, outputVal); //L
		overwriteValue(x, 0, outputVal, val);
		val = G * value(x, SIZE - 2, outputVal); //R
		overwriteValue(x, SIZE - 1, outputVal, val);
		val = G * value(SIZE - 2, x, outputVal); //B
		overwriteValue(SIZE - 1, x, outputVal, val);

	}


	//corners
	val = G * value(1, 0, outputVal); 
	overwriteValue(0, 0, outputVal, val);
	val = G * value(SIZE - 2, 0, outputVal); 
	overwriteValue(SIZE -1, 0, outputVal, val);
	val = G * value(0, SIZE - 2, outputVal); 
	overwriteValue(0, SIZE - 1, outputVal, val);
	val = G * value(SIZE - 1, SIZE - 2, outputVal); 
	overwriteValue(SIZE - 1, SIZE - 1, outputVal, val);

	/*for (int x = 0; x < 4; x++) {
		for (int y = 0; y < 4; y++) {
			printf("(%d,%d): %f,\t", x, y, value(x, y, outputVal));
		}
		printf("\n");
	}*/
	printf("(2,2): %f,\n", value(2, 2, outputVal));
}

int main(int argc, char** argv)
{
	printf("Size of grid: %d nodes\n", TOTAL_S);

	int iterations = atoi(argv[1]);

	float* startValues;
	startValues = (float*)malloc(sizeof(float) * TOTAL_S);
	for (int i = 0; i < TOTAL_S; i++) {
		startValues[i] = 0.0; //init all values at 0
	}

	

	if (PARALLEL == 0) { //sequent

		float* elements4;
		float* outputValSeq;
		float* elements3 ; //num 10
		elements3 = (float*)malloc(sizeof(float) * TOTAL_S);
		elements4 = (float*)malloc(sizeof(float) * TOTAL_S);
		outputValSeq = (float*)malloc(sizeof(float) * TOTAL_S);
		for (int i = 0; i < TOTAL_S; i++) {
			if (i == 10) {
				elements3[i] = 1.0; //init all values at 0
			}
			else {
				elements3[i] = 0.0; //init all values at 0
			}
			elements4[i] = 0.0; //init all values at 0
			outputValSeq[i] = 0.0;
		}


		/*for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				printf("(%d, %d, %d)\t",elements3[index(x,y)], elements4[index(x, y)], outputValSeq[index(x, y)] );
			}
			printf("\n");
		}*/ 

		for (int i = 0; i < iterations; i++) {
			iterateSeq(outputValSeq, elements3, elements4, i);
			float* temp = elements4;
			elements4 = elements3;
			elements3 = outputValSeq;
			outputValSeq = temp;
		}
	}
	else if (PARALLEL == 1) { //parallel

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
		overwriteValue(2, 2, startValues, 1.0); //drum was hit
		cudaMemcpy(elements1, startValues, TOTAL_S * sizeof(float), cudaMemcpyHostToDevice);
		dim3 Block(4, 4, 1);
		dim3 Grid(1, 1, 1);
		for (int i = 0; i < iterations; i++) {

			iterate << <Grid, Block >> > (outputVal, elements1, elements2);

			float* temp = elements2;
			elements2 = elements1;
			elements1 = outputVal;
			outputVal = temp;
		}

		//// copy to CPU
		cudaMemcpy(startValues, elements1, TOTAL_S * sizeof(float), cudaMemcpyDeviceToHost);
		free(startValues);
	}

    return 0;
}
