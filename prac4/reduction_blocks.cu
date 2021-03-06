// -*- mode: c++; -*-

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

void reduction_gold(float* odata, float* idata, const unsigned int len) 
{
  *odata = 0;
  for(int i=0; i<len; i++) *odata += idata[i];
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__device__ float d_sum = 0.0f;

__device__ int lock = 0;

__global__ void reduction(float *data)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // first, each thread loads data for block into shared memory
    temp[tid] = data[tid + bid*blockDim.x];

    // next, we perform binary tree reduction

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  temp[tid] += temp[tid+d];
    }

    // finally, first thread puts result into global memory
    if(tid == 0)
    {
      do {} while (atomicCAS(&lock, 0, 1));
      
      d_sum += temp[0];
      __threadfence();
      lock = 0;
    }
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_elements, num_blocks, num_threads, mem_size, shared_mem_size;

  float *h_data, cpu_sum, h_sum;
  float *d_idata;

  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 512;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

  // compute reference solutions
  reduction_gold(&cpu_sum, h_data, num_elements);

  // allocate device memory input array
  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );

  // copy host memory to device input array
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel
  num_blocks   = 4;
  num_threads  = 128;
  shared_mem_size = sizeof(float) * num_elements;
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_idata);
  cudaThreadSynchronize();
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host
  checkCudaErrors( cudaMemcpyFromSymbol(&h_sum, d_sum, sizeof(float)) );

  // check results
  //printf("reduction error = %f\n",h_data[0]-sum);
  printf("reduction error = %f\n", h_sum - cpu_sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
