#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

#define TEST_THREAD 0 //1 if you want to test runtimes for set 1,2,4,8,16,32,128,256

#define MAX_MSE 0.00001f
#define MAX_THREAD 1024

__global__ void rectify(unsigned char* d_input_img, unsigned char* d_output_img, unsigned int size, const int threadNum)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += threadNum) {
		if (d_input_img[i] < 127)
			d_output_img[i] = 127;
		else
			d_output_img[i] = d_input_img[i];
	}
}

float get_MSE(char* input_filename_1, char* input_filename_2)
{
	unsigned error1, error2;
	unsigned char* image1, * image2;
	unsigned width1, height1, width2, height2;

	error1 = lodepng_decode32_file(&image1, &width1, &height1, input_filename_1);
	error2 = lodepng_decode32_file(&image2, &width2, &height2, input_filename_2);
	if (error1) printf("error %u: %s\n", error1, lodepng_error_text(error1));
	if (error2) printf("error %u: %s\n", error2, lodepng_error_text(error2));
	if (width1 != width2) printf("images do not have same width\n");
	if (height1 != height2) printf("images do not have same height\n");

	// process image
	float im1, im2, diff, sum, MSE;
	sum = 0;
	for (int i = 0; i < width1 * height1; i++) {
		im1 = (float)image1[i];
		im2 = (float)image2[i];
		diff = im1 - im2;
		sum += diff * diff;
	}
	MSE = sqrt(sum) / (width1 * height1);

	free(image1);
	free(image2);

	return MSE;
}

int main(int argc, char* argv[])
{
	unsigned error;
	unsigned char* host_input_img, * host_output_img;
	unsigned char* device_input_imgg;
	unsigned width, height;
	int input_img_size, output_img_size;
	GpuTimer gTim;

	// Decode png image
	char* input_filename = argv[1];
	char* output_filename = argv[2];
	unsigned int threadNum = atoi(argv[3]);

	if (threadNum > MAX_THREAD) threadNum = MAX_THREAD; //cant be more than 1024 threads

	error = lodepng_decode32_file(&host_input_img, &width, &height, input_filename);
	input_img_size = width * height * 4 * sizeof(unsigned char);

	// Allocate device memory for the output image
	unsigned char* c_image, * c_new_image;
	cudaMalloc((void**)&c_image, input_img_size);
	cudaMalloc((void**)&c_new_image, input_img_size);

	//copy memory to gpu
	cudaMemcpy(c_image, host_input_img, input_img_size, cudaMemcpyHostToDevice);


	if (TEST_THREAD) { //0 test threads with time and no saving the picture

		// Number of threads to test and allocate mem to store elapsed times and timer
		int test_number_threads[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

		for (int i = 0; i < 11; i++)
		{
			double time_used = 0;

			printf("Running rectify on %s.\n", input_filename);
			printf("There are %d threads running...\n", test_number_threads[i]);

			gTim.Start();
			rectify << <1, test_number_threads[i] >> > (c_image, c_new_image, input_img_size, test_number_threads[i]);
			cudaDeviceSynchronize();
			gTim.Stop();

			time_used = gTim.Elapsed();

			printf("Elapsed time: %f \n", time_used);
			//printf("%d, %f \n", test_number_threads[i], time_used);
		}
	}
	else { //1 iteration normally

		printf("Running rectify on %s.\n", input_filename);
		printf("There are %d threads running...\n", threadNum);

		gTim.Start();
		rectify << <1, threadNum >> > (c_image, c_new_image, input_img_size, threadNum);
		cudaDeviceSynchronize();
		gTim.Stop();

		printf("Elapsed time: %f \n", gTim.Elapsed());

	}

	unsigned char* device_output_img = (unsigned char*)malloc(input_img_size);
	cudaMemcpy(device_output_img, c_new_image, input_img_size, cudaMemcpyDeviceToHost);
	cudaFree(c_image);
	cudaFree(c_new_image);

	lodepng_encode32_file(output_filename, device_output_img, width, height);

	free(device_output_img);

	//// get mean squared error between image1 and image2
	//char* input_filename_1 = "Test_1_pooled.png";
	//char* input_filename_2 = "pooled1.png";

	//float MSE = get_MSE(input_filename_1, input_filename_2);

	//if (MSE < MAX_MSE) {
	//	printf("Images are equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
	//}
	//else {
	//	printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
	//}
	//return 0;
}