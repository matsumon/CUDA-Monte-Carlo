
// Array multiplication: C = A * B:

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK		64	// number of threads in each block
#endif

#ifndef DATASET_SIZE
#define DATASET_SIZE			( 8*1024*1024 )	// size of the array
#endif

float hA[DATASET_SIZE];
float hB[DATASET_SIZE];
float hC[DATASET_SIZE];

#ifndef TOL
#define TOL		0.00001f		// tolerance to relative error
#endif

void
CudaCheckError()
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
	}
}

// array multiplication on the device: C = A * B

__global__  void ArrayMul(float* A, float* B, float* C)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < DATASET_SIZE)
		C[gid] = A[gid] * B[gid];
}



// main program:

int
main(int argc, char* argv[])
{
	int dev = findCudaDevice(argc, (const char**)argv);

	// fill host memory:

	for (int i = 0; i < DATASET_SIZE; i++)
	{
		hA[i] = hB[i] = sqrtf((float)i);
	}

	// allocate device memory:

	float* dA, * dB, * dC;

	cudaMalloc((void**)(&dA), sizeof(hA));
	cudaMalloc((void**)(&dB), sizeof(hB));
	cudaMalloc((void**)(&dC), sizeof(hC));
	CudaCheckError();

	// copy host memory to the device:

	cudaMemcpy(dA, hA, DATASET_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, DATASET_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	CudaCheckError();

	// setup the execution parameters:

	dim3 grid(DATASET_SIZE / THREADS_PER_BLOCK, 1, 1);
	dim3 threads(THREADS_PER_BLOCK, 1, 1);

	// create and start the timer:

	cudaDeviceSynchronize();

	// allocate the events that we'll use for timing:

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	CudaCheckError();

	// record the start event:

	cudaEventRecord(start, NULL);
	CudaCheckError();

	// execute the kernel:

	ArrayMul << < grid, threads >> > (dA, dB, dC);

	// record the stop event:

	cudaEventRecord(stop, NULL);
	CudaCheckError();

	// wait for the stop event to complete:

	cudaEventSynchronize(stop);
	CudaCheckError();

	float msecTotal;
	cudaEventElapsedTime(&msecTotal, start, stop);
	CudaCheckError();

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double multsPerSecond = (double)DATASET_SIZE / secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	fprintf(stderr, "%12d\t%4d\t%10.2lf\n", DATASET_SIZE, THREADS_PER_BLOCK, megaMultsPerSecond);

	// copy result from the device to the host:

	cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
	CudaCheckError();

	// check for correctness:

	fprintf(stderr, "Checking computed result for correctness:\n");
	bool correct = true;

	for (int i = 1; i < DATASET_SIZE; i++)
	{
		double error = ((double)hC[i] - (double)i) / (double)i;
		if (fabs(error) > TOL)
		{
			fprintf(stderr, "C[%12d] = %10.2lf, correct = %10.2lf\n", i, (double)hC[i], (double)i);
			correct = false;
		}
	}

	if (correct)
		fprintf(stderr, "PASS.\n");
	else
		fprintf(stderr, "FAIL.\n");

	// clean up:

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	CudaCheckError();

	return 0;
}

