
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "gputimer.h"
#include "wm.h"

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <math.h>

// Set TEST to 1 in order to run convolution for different number of threads.
#define TEST 0
//#define THREADS_PER_BLOCK 1024

// Kernel function
__global__ void convolutionKernel(unsigned char* d_input_img, unsigned char* d_output_img, float** d_w, unsigned input_width,
	unsigned input_height, int n_threads, int max_index)
{
	// out_i is index in output array, in_i is index in input array
	int out_i = threadIdx.x + blockIdx.x * blockDim.x;
	int in_i = 0;
	unsigned out_width = input_width - 2;

	while (out_i < max_index)
	{
		int out_row = out_i / out_width;
		in_i = (out_row + 1) * input_width + (out_i % out_width) + 1;

		// Result from convolution for each pixel channel
		float sumR = 0;
		float sumG = 0;
		float sumB = 0;

		for (int i = -1; i <= 1; i++)    // Iterate through rows and columns of the convolution matrix
		{
			for (int j = -1; j <= 1; j++)
			{
				sumR += d_w[1 + i][1 + j] * d_input_img[4 * (in_i + input_width * i + j) + 0];
				sumG += d_w[1 + i][1 + j] * d_input_img[4 * (in_i + input_width * i + j) + 1];
				sumB += d_w[1 + i][1 + j] * d_input_img[4 * (in_i + input_width * i + j) + 2];
			}
		}

		sumR = roundf(sumR);
		sumG = roundf(sumG);
		sumB = roundf(sumB);

		d_output_img[out_i * 4 + 0] = (sumR > 255) ? 255 : ((sumR < 0) ? 0 : (unsigned char)sumR);
		d_output_img[out_i * 4 + 1] = (sumG > 255) ? 255 : ((sumG < 0) ? 0 : (unsigned char)sumG);
		d_output_img[out_i * 4 + 2] = (sumB > 255) ? 255 : ((sumB < 0) ? 0 : (unsigned char)sumB);
		d_output_img[out_i * 4 + 3] = d_input_img[in_i * 4 + 3];		//Do not change Alpha component

		out_i += n_threads;
	}
}

int main(int argc, char* argv[])
{
	// Command line inputs
	char* input_filename = argv[1];
	char* output_filename = argv[2];
	size_t n_threads = atoi(argv[3]);

	/*char* input_filename = "Test_1.png";
	char* output_filename = "Test_1_out.png";
	size_t n_threads = 512;*/

	// Host/Device image variables
	unsigned error;
	unsigned char* host_input_img, * host_output_img;
	unsigned char* device_input_img, * device_output_img;
	unsigned input_width, input_height;
	unsigned output_width, output_height;
	size_t input_img_size, output_img_size;

	float** d_w;		// Device convolution matrix pointer
	size_t n_blocks;
	GpuTimer timer{};

	// Decode png image
	error = lodepng_decode32_file(&host_input_img, &input_width, &input_height, input_filename);
	output_width = input_width - 2;
	output_height = input_height - 2;
	input_img_size = input_width * input_height * 4 * sizeof(unsigned char);
	output_img_size = output_width * output_height * 4 * sizeof(unsigned char);

	// We need #blocks = #threads / #threads per block
	//n_blocks = (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	n_blocks = 1;

	// Allocate host memory for the output image
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	host_output_img = (unsigned char*)malloc(output_img_size);

	// Allocate device memory for input and output images
	cudaMalloc((void**)&device_input_img, input_img_size);
	cudaMalloc((void**)&device_output_img, output_img_size);

	// Allocate device memory for convolution matrix
	cudaMalloc((void**)&d_w, 3 * sizeof(float*));
	float* h_w[3];
	for (int i = 0; i < 3; i++)
	{
		cudaMalloc((void**)&h_w[i], 3 * sizeof(float));
		cudaMemcpy(h_w[i], w[i], 3 * sizeof(float), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(d_w, h_w, 3 * sizeof(float*), cudaMemcpyHostToDevice);

	//Copy data from host to device
	cudaMemcpy(device_input_img, host_input_img, input_img_size, cudaMemcpyHostToDevice);

	// Test (run in loop to compute timers for different #threads)
	if (TEST)
	{
		// Number of threads to test and allocate mem to store elapsed times and timer
		int test_number_threads[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

		for (int i = 0; i < 11; i++)
		{
			timer.Start();
			n_threads = test_number_threads[i];

			// Launch kernel
			//printf("Running convolution on %s.\n", input_filename);
			//printf("There are %d threads running.\n", n_threads);
			convolutionKernel << <n_blocks, n_threads >> > (device_input_img, device_output_img, d_w, input_width, input_height,
				n_threads, output_width * output_height);
			cudaDeviceSynchronize();

			timer.Stop();
			printf("%d %f\n", n_threads, timer.Elapsed());
		}
	}

	// Normal execution (takes #threads from command line)
	else {
		timer.Start();

		// Launch kernel
		printf("Running convolution on \"%s\".\n", input_filename);
		printf("There are %d threads running.\n", n_threads);
		convolutionKernel << <n_blocks, n_threads >> > (device_input_img, device_output_img, d_w, input_width, input_height,
			n_threads, output_width * output_height);
		cudaDeviceSynchronize();

		timer.Stop();
		printf("Elapsed time: %f milliseconds\n", timer.Elapsed());
	}

	// Copy output data from device to host
	cudaMemcpy(host_output_img, device_output_img, output_img_size, cudaMemcpyDeviceToHost);

	// Encode output image
	lodepng_encode32_file(output_filename, host_output_img, output_width, output_height);

	// Cleanup
	free(host_input_img);  free(host_output_img);
	cudaFree(device_input_img); cudaFree(device_output_img);

	// Free device convolution matrix
	for (int i = 0; i < 3; i++)
		cudaFree(h_w[i]);
	cudaFree(d_w);

	return 0;
}