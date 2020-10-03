#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

// Set TEST to 1 in order to run pooling for different total numbers of threads.
#define TEST 0

// Returns max value from array (only callable by device, host cannot call)
__device__ unsigned char max(unsigned char arr[], int size)
{
	unsigned char max = arr[0];

	for (int i = 1; i < size; i++)
		if (arr[i] > max) max = arr[i];

	return max;
}

// Kernel function
__global__ void poolKernel(unsigned char* d_input_img, unsigned char* d_output_img, const int pools_per_thread,
	const int out_width, const int out_height)
{
	// out_i = index for the output img array, in_i = index for input img array
	int out_i = (threadIdx.x + blockIdx.x * blockDim.x) * pools_per_thread;
	int in_i;

	// We advance through the output array pixel by pixel (4 chars)
	for (int j = 0; j < pools_per_thread && out_i < out_height * out_width; j++, out_i++)
	{
		// Formula to transform index in output img to index in input img
		in_i = (out_i / out_width) * (out_width * 4) + (out_i % out_width) * 2;

		// The values in array correspond to the squares at bottom left, bottom right, top left and top right respectively
		unsigned char valR[4] = { d_input_img[4 * in_i] , d_input_img[4 * in_i + 4],
			d_input_img[4 * in_i + out_width * 2 * 4], d_input_img[4 * in_i + 4 + out_width * 2 * 4] };
		unsigned char valG[4] = { d_input_img[4 * in_i + 1] , d_input_img[4 * in_i + 5] ,
			d_input_img[4 * in_i + 1 + out_width * 2 * 4], d_input_img[4 * in_i + 5 + out_width * 2 * 4] };
		unsigned char valB[4] = { d_input_img[4 * in_i + 2] , d_input_img[4 * in_i + 6] ,
			d_input_img[4 * in_i + 2 + out_width * 2 * 4], d_input_img[4 * in_i + 6 + out_width * 2 * 4] };
		unsigned char valA[4] = { d_input_img[4 * in_i + 3] , d_input_img[4 * in_i + 7] ,
			d_input_img[4 * in_i + 3 + out_width * 2 * 4], d_input_img[4 * in_i + 7 + out_width * 2 * 4] };

		// Assign max values among the 4 squares to the output
		d_output_img[4 * out_i] = max(valR, 4);
		d_output_img[4 * out_i + 1] = max(valG, 4);
		d_output_img[4 * out_i + 2] = max(valB, 4);
		d_output_img[4 * out_i + 3] = max(valA, 4);
	}
}

int main(int argc, char* argv[])
{
	unsigned error;
	unsigned char* host_input_img, * host_output_img;
	unsigned char* device_input_img, * device_output_img;
	unsigned width, height;
	int input_img_size, output_img_size;
	GpuTimer timer{};

	char* input_filename = argv[1];
	char* output_filename = argv[2];
	int number_threads = atoi(argv[3]);

	// Decode png image
	error = lodepng_decode32_file(&host_input_img, &width, &height, input_filename);
	input_img_size = width * height * 4 * sizeof(unsigned char);
	output_img_size = width / 2 * height / 2 * 4 * sizeof(unsigned char);

	// Allocate host memory for the output image
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	host_output_img = (unsigned char*)malloc(output_img_size);

	// Allocate device memory for input and output images
	cudaMalloc((void**)&device_input_img, input_img_size);
	cudaMalloc((void**)&device_output_img, output_img_size);

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

			// Launch kernel
			printf("There are %d threads running.\n", test_number_threads[i]);
			poolKernel << <1, test_number_threads[i] >> > (device_input_img, device_output_img, width * height / (4 * test_number_threads[i]),
				width / 2, height / 2);
			cudaDeviceSynchronize();

			timer.Stop();
			printf("Elapsed time: %f milliseconds\n", timer.Elapsed());
		}
	}

	// Normal execution (takes #threads from command line)
	else {
		timer.Start();

		// Launch kernel
		printf("Running pooling on %s.\n", input_filename);
		printf("There are %d threads running.\n", number_threads);
		poolKernel << <1, number_threads >> > (device_input_img, device_output_img, width * height / (4 * number_threads),
			width / 2, height / 2);
		cudaDeviceSynchronize();

		timer.Stop();
		printf("Elapsed time: %f milliseconds\n", timer.Elapsed());
	}

	// Copy output data from device to host
	cudaMemcpy(host_output_img, device_output_img, output_img_size, cudaMemcpyDeviceToHost);

	// Encode output image
	lodepng_encode32_file(output_filename, host_output_img, width / 2, height / 2);

	// Cleanup
	free(host_input_img);  free(host_output_img);
	cudaFree(device_input_img); cudaFree(device_output_img);

	return 0;
}