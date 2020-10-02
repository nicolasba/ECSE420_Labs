#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>



__global__ void rectify(unsigned char* d_input_img, unsigned char* d_output_img, unsigned int size)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < size) {
		if (d_input_img[index] < 127) {
			d_output_img[index] = 127;
		}
		else {
			d_output_img[index] = d_input_img[index];
		}
	}
}

int main(int argc, char* argv[])
{
	unsigned error;
	unsigned char* host_input_img, * host_output_img;
	unsigned char* device_input_imgg;
	unsigned width, height;
	int input_img_size, output_img_size;

	// Decode png image
	char* input_filename = argv[1];
	char* output_filename = argv[2];
	unsigned int threadNum = atoi(argv[3]);
	error = lodepng_decode32_file(&host_input_img, &width, &height, input_filename);
	input_img_size = width * height * 4 * sizeof(unsigned char);
	

	// Allocate host memory for the output image
	unsigned char* c_image, * c_new_image;
	cudaMalloc((void**)&c_image, input_img_size);
	cudaMalloc((void**)&c_new_image, input_img_size);

	//copy memory to gpu
	cudaMemcpy(c_image, host_input_img, input_img_size, cudaMemcpyHostToDevice);

	unsigned int blockNum = (input_img_size + threadNum - 1) / threadNum;

	rectification << < blockNum, threadNum >> > (c_image, c_new_image, input_img_size);


	unsigned char* device_output_img = (unsigned char*)malloc(size_image);
	cudaMemcpy(device_output_img, c_new_image, input_img_size, cudaMemcpyDeviceToHost);
	cudaFree(c_image);
	cudaFree(c_new_image);


	lodepng_encode32_file(output_filename, device_output_img, width, height);

	
	return 0;
}