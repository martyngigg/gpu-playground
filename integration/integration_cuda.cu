// -*- mode: c++; -*-
//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "CUDATimer.h"
#include "Workspace2D.h"

#include <iostream>

namespace sandbox
{
  Workspace2DPtr integrate_with_cuda(const Workspace2DPtr & workspace)
  {
    Workspace2D * result = new Workspace2D(workspace->nVectors, 1);
    std::cout << "Running integration using CUDA...\n";
        
    CUDATimer wallClock;

    std::cout << "Finished in " << wallClock.elapsed() << " s\n";
    return Workspace2DPtr(result);
  }
}



