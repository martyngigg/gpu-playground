// -*- mode: c++; -*-

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// device variable and kernel
__device__ float d_test = 0.0f;
__global__ void kernel1() { d_test = 1.0; }

int main() {

  // initialise variables
  float h_test = 0.0;
  //cudaMemcpyToSymbol(d_test,&h_test, sizeof(float));

  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("cudaMemset error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  // invoke kernel
  kernel1 <<<1,1>>> ();

  cudaThreadSynchronize();

  // check for error
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("kernel error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  // Copy device variable to host and print
  cudaMemcpyFromSymbol(&h_test,d_test,sizeof(float));

  printf("Value of h_test: %f\n:", h_test);

}
