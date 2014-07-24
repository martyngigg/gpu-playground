//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "CUDATimer.h"
#include "Workspace2D.h"

#include <iostream>

namespace sandbox
{
  void integrate_with_cuda(const size_t nvec, const size_t npts)
  {
    std::cout << "Running integration using CUDA...\n";
    CUDATimer wallClock;

    std::cout << "Finished in " << wallClock.elapsed() << " s\n";
  }
}



