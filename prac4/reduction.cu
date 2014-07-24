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

__global__ void reduction(float *g_odata, float *g_idata, unsigned int ndata)
{
    // dynamically allocated shared memory

    extern  __shared__  float temp[];

    int tid = threadIdx.x;

    // first, each thread loads data into shared memory

    temp[tid] = g_idata[tid];
    bool active = (tid < ndata);
    // next, we perform binary tree reduction

    //for (int d = blockDim.x>>1; d > 0; d >>= 1) {

    int delta = ndata - blockDim.x;
    printf("delta = %d\n", delta);
    int d0 = 1 << (31 - __clz(((int)blockDim.x-1)));
    for( int d = d0; d > 0; d >>= 1) {
      printf("d = %d\n", d);
      __syncthreads();  // ensure previous step completed 
      if (active && tid < d)
      { 
        int mirror =  tid+d;
        temp[tid] += temp[mirror];
        __syncthreads();  // ensure previous step completed 
        if (delta > 0)
        {
          printf("extra idx = %d\n", mirror + delta);
          temp[tid] += temp[mirror + delta];
        }
      }
    }

    // finally, first thread puts result into global memory

    if (tid==0) g_odata[0] = temp[0];
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_elements, num_threads, mem_size, shared_mem_size;

  float *h_data, sum;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 512;
  num_threads  = num_elements;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

  // compute reference solutions
  reduction_gold(&sum, h_data, num_elements);

  // allocate device memory input and output arrays
  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, sizeof(float)) );

  // copy host memory to device input array
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel
  shared_mem_size = sizeof(float) * num_elements;
  reduction<<<1, num_threads, shared_mem_size>>>(d_odata,d_idata, num_elements);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, sizeof(float),
                              cudaMemcpyDeviceToHost) );

  // check results
  printf("CPU sum = %f\n", sum);
  printf("GPU sum = %f\n", h_data[0]);
  printf("GPU reduction error = %f\n",h_data[0]-sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
