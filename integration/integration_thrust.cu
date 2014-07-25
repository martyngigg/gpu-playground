// -*- mode: c++; -*-
//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "CUDATimer.h"
#include "Workspace2D.h"

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <iostream>

namespace sandbox
{

  using thrust::adjacent_difference;
  using thrust::device_vector;
  using thrust::inner_product;

  Workspace2DPtr integrate_with_thrust(const Workspace2DPtr & workspace)
  {
    const size_t nVectors = workspace->nVectors;
    Workspace2D * result = new Workspace2D(nVectors, 1);
    std::cout << "Running integration using Thrust...\n";
    CUDATimer wallClock;

    #pragma omp parallel for
    for(size_t i = 0; i < nVectors; ++i)
    {
      device_vector<DataType> d_xIn = workspace->dataY[i];
      device_vector<DataType> d_yIn = workspace->dataX[i];
      device_vector<DataType> widths(workspace->nPts - 1);
      adjacent_difference(d_xIn.begin(), d_xIn.end(), widths.begin());
      result->dataY[i][0] = inner_product(d_yIn.begin(), d_yIn.end(), widths.begin(),
					  0.0);
    }

    std::cout << "Finished in " << wallClock.elapsed() << " s\n";
    return Workspace2DPtr(result);
  }
}



