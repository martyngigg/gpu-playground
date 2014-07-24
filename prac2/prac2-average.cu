// -*- mode: c++; -*-

////////////////////////////////////////////////////////////////////////
// GPU averge of normalised distribution
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////

__constant__ float A, B, C;
__constant__ int NVALS_PER_THREAD;

////////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////////

__global__ void quadratic(float *z, float *v)
{
  float avg(0.0f), zval(0.0f);
  z = z + threadIdx.x + blockDim.x*blockIdx.x;
  for(int i = 0; i < NVALS_PER_THREAD; ++i)
  {
    z += 1;
    zval = *z;
    avg += A*zval*zval + B*zval + C;
  }
  v = v + threadIdx.x + blockDim.x*blockIdx.x;
  *v = avg/NVALS_PER_THREAD;
}


int main(int argc, const char **argv)
{
  // initialize card
  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // equation constants
  const float a(5.0), b(3.0), c(10.0);

  // GPU sizes
  const int nthreads_per_block(512), nblocks(4);
  const int navgs_per_thread(5000);

  const int nthreads = nthreads_per_block * nblocks;
  const int nrand = nthreads*navgs_per_thread;
  // allocate memory on host and device
  float *hostValues = (float *)malloc(sizeof(float)*nthreads);
  float *devRand, *devValues;
  checkCudaErrors( cudaMalloc((void **)&devRand, sizeof(float)*nrand) );
  checkCudaErrors( cudaMalloc((void **)&devValues, sizeof(float)*nthreads) );

  // copy constants
  checkCudaErrors( cudaMemcpyToSymbol(A,    &a,    sizeof(a)) );
  checkCudaErrors( cudaMemcpyToSymbol(B,    &b,    sizeof(b)) );
  checkCudaErrors( cudaMemcpyToSymbol(C,    &c,    sizeof(c)) );
  checkCudaErrors( cudaMemcpyToSymbol(NVALS_PER_THREAD, &navgs_per_thread, 
                                      sizeof(navgs_per_thread)) );

  // Initialize random number generator
  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateNormal(gen, devRand, nrand, 0.0f, 1.0f) );

  // Call kernel
  cudaEventRecord(start);

  quadratic<<<nblocks, nthreads_per_block>>>(devRand, devValues);
  getLastCudaError("quadratic execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  //Copy values back to host
  checkCudaErrors( cudaMemcpy(hostValues,devValues,nthreads*sizeof(float), cudaMemcpyDeviceToHost) );  

  // Average values array
  float average = 0.0f;
  for(int i = 0; i < nthreads; ++i)
  {
    average += hostValues[i];
  }
  average = average/nthreads;
  printf("Computed average using %d threads: %f\n", nthreads, average);
  printf("Total time: %fms\n", milli);

  checkCudaErrors( curandDestroyGenerator(gen) );
  free(hostValues);
  checkCudaErrors( cudaFree(devRand) ); 
  checkCudaErrors( cudaFree(devValues) ); 

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();
}
