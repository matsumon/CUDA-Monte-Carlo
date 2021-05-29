# CUDA-Monte-Carlo
Introduction

Monte Carlo simulation is used to determine the range of outcomes for a series of parameters, each of which has a probability distribution showing how likely each option is to happen. In this project, you will take the Project #1 scenario and develop a Monte Carlo simulation of it, determining how likely a particular output is to happen.

The Scenario

Use the same scenario from Project #1.
Using Linux for this Project

On both rabbit and the DGX system, here is a working Makefile:


CUDA_PATH	=	/usr/local/apps/cuda/cuda-10.1
CUDA_BIN_PATH	=	$(CUDA_PATH)/bin
CUDA_NVCC	=	$(CUDA_BIN_PATH)/nvcc

montecarlo:	montecarlo.cu
		$(CUDA_NVCC) -o montecarlo  montecarlo.cu

Before you use the DGX, do your development on the rabbit system (Slide #3 of the DGX noteset). It is a little friendlier because you don't have to run your program through a batch submission. But, don't take any final performance numbers from rabbit, just get your program running there.
But, if you decide to use Visual Studio on your own machine, you must first install the CUDA Toolkit!

If you are trying to run CUDA on your own Visual Studio system, make sure your machine has the CUDA toolkit installed. It is available here: https://developer.nvidia.com/cuda-downloads

Running CUDA in Visual Studio

Get the NewCudaArrayMul2019.zip file. Un-zip the file and double-click on the .sln file. This is the CUDA version of an array multiply program. Modify it for this assignment.

Requirements:

    The ranges are:
    Variable	Meaning	Range
    g	Ground distance to the cliff face	20. - 30.
    h	Height of the cliff face	10. - 40.
    d	Upper deck distance to the castle	10. - 20.
    v	Cannonball initial velocity	30. - 50.
    θ	Cannon firing angle	70. - 80.

    Note: these are not the same numbers as we used before!

    Run this for at least four BLOCKSIZEs (i.e., the number of threads per block) of 16, 32, 64, and 128, combined with NUMTRIALS sizes of at least 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, and 1M.

    Be sure the NUMTRIALS are in multiples of 1024, that is, for example, for "32K" use 32,768, not 32,000.

    Record timing for each combination. For performance, use some appropriate units like MegaTrials/Second.

    For this one, use CUDA timing, not OpenMP timing.

    Do a table and two graphs:
        Performance vs. NUMTRIALS with multiple curves of BLOCKSIZE
        Performance vs. BLOCKSIZE with multiple curves of NUMTRIALS 

    Like Project #1 before, fill the arrays ahead of time with random values. Send them to the GPU where they can be used as look-up tables. 
