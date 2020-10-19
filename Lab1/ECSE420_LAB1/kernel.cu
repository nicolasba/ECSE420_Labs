
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
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


char computeOutput(int values[]) {
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

	char toReturn = compute + '0';
	return toReturn;
}

int main(int argc, char* argv[])
{
	char* input_filename = argv[1];
	int fileLines = atoi(argv[2]);
	char* output_filename = argv[3];

	FILE* file = fopen(input_filename, "r"); //read
	FILE* outFile = fopen(output_filename, "w"); //write from scratch

	char answer = NULL;
	char line[7];

	clock_t start = clock();


	while (fgets(line, sizeof(line), file)) {
		//printf("%s\n", line);
		char* copy = strdup(line);
		int ops[3] = { 0,0,0 };
		char* token = strtok(copy, ",");
		ops[0] = atoi(token);
		//printf("%d\t", ops[0]);
		token = strtok(NULL, ",");
		ops[1] = atoi(token);
		//printf("%d\t", ops[1]);
		token = strtok(NULL, ",");
		ops[2] = atoi(token);
		//printf("%d\n", ops[2]);
		answer = computeOutput(ops);
		//printf("%c\n", answer);
		fprintf(outFile, "%c\n", answer);
	}

	clock_t stop = clock();
	int totalTime = (stop -start)*1000/ CLOCKS_PER_SEC;

	fclose(outFile);
	fclose(file);
	printf("Sequential Logic Complete Time Taken: %d (s) %d (ms)\n", totalTime / 1000, totalTime % 1000); //time in seconds
	//compareFiles("out_seq_10000.txt", "sol_10000.txt");
	return 0;
}

