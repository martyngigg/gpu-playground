// -*- mode: c++; -*-
//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  
  x[tid] = (float) threadIdx.x;
}

__global__ void sum_vectors(float *a, float *b, float * c)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  c[tid] = a[tid] + b[tid];
}


//
// main code
//

int main(int argc, const char **argv)
{
  float *ha_x, *hb_x, *hc_x, *da_x, *db_x, *dc_x;
  int   nblocks, nthreads, nsize, n; 
  
  // initialise card

  int stat = findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;

  nsize = nblocks*nthreads;

  // allocate memory for host input arrays
  ha_x = (float *)malloc(nsize*sizeof(float));
  hb_x = (float *)malloc(nsize*sizeof(float));
  for(int i = 0; i < nsize; ++i)
  {
    ha_x[i] = 2.0;
    hb_x[i] = 4.0;
  }

  checkCudaErrors(cudaMalloc((void **)&da_x, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&db_x, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&dc_x, nsize*sizeof(float)));

  // execute kernel
  
  sum_vectors<<<nblocks,nthreads>>>(da_x,db_x,dc_x);
  getLastCudaError("Kernel execution failed\n");

  // copy back results and print them out

  hc_x = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors( cudaMemcpy(hc_x,dc_x,nsize*sizeof(float),
                   cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,hc_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(da_x));
  free(ha_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
