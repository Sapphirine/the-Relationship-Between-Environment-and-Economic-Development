#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pycuda import driver as cuda
from pycuda import compiler, gpuarray, tools
from pycuda.compiler import SourceModule
import time
import pandas as pd
import csv

# -- initialize the device
import pycuda.autoinit

mod = SourceModule("""

# define BLOCK_SIZE 32

__global__ void parallelSum(float * input, float * output, int len)
{
    __shared__ float partialSum[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x*blockDim.x*2;

    if((start+t)<len)
    {
        partialSum[t] = input[start+t];
        if(start+t+blockDim.x < len)
            partialSum[blockDim.x+t] = input[start+t+blockDim.x];
        else
            partialSum[blockDim.x+t] =0;
    }
    else
    {
        partialSum[t] = 0;
        partialSum[blockDim.x+t] = 0;

    }
    __syncthreads();

    for(unsigned int stride = blockDim.x; stride >0; stride/=2){
        __syncthreads();
        if(t<stride)
            partialSum[t]+=partialSum[t+stride];
    }

    output[blockIdx.x] = partialSum[0];
}
""")


data = pd.read_csv('raw_data.csv')
raw_data = data.values[:, 2]

prcsed_data = []
time_GPU = []
start = time.time()
for i in range(47*16):

    a = raw_data[8760*i: 8760*(i + 1)].astype(np.float32)
    l = len(a)
    l = l.astype(np.int32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    parallelSum = mod.get_function("parallelSum")
    parallelSum(a_gpu, b_gpu, l, block=(1024, 1, 1))
    b = np.empty_like(a)
    cuda.memcpy_dtoh(b, b_gpu)
    prcsed_data[i] = b

time_GPU = time.time() - start

csvfile = file('data.csv', 'wb')
writer = csv.writer(csvfile)
writer.writerows(prcsed_data)
csvfile.close()

print time_GPU

