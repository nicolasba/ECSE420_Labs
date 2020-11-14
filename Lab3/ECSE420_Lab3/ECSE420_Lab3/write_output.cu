#include "write_output.h"

#include <stdio.h>
#include <stdlib.h>

int write_output(int* output, int length, char* filepath) {

	FILE* fp = fopen(filepath, "w");
	if (fp == NULL) {
		fprintf(stderr, "Couldn't open file for reading\n");
		exit(1);
	}

	fprintf(fp, "%d\n", length);
	for (int i = 0; i < length; i++)
		fprintf(fp, "%d\n", output[i]);

	fclose(fp);
	return 0;
}