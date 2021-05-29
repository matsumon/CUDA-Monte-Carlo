#!/bin/bash
#SBATCH  -J  Five
#SBATCH  -A  cs475-575
#SBATCH  -p  class
#SBATCH  --gres=gpu:1
#SBATCH  -o  five.out
#SBATCH  -e  five.err
for t in  128
do
    for s in  524288 1048576
    do
        /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$t  -DNUMTRIALS=$s -o five five.cu
        ./five
    done
done