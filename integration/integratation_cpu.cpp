//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "HostTimer.h"
#include "Workspace2D.h"

#include <iostream>

namespace sandbox
{
  void integrate_on_cpu(const size_t nvec, const size_t npts)
  {
    std::cout << "Running integration on CPU...\n";
    HostTimer wallClock;

    auto workspace = createCosineWorkspace(nvec, npts);

    std::cout << "Finished in " << wallClock.elapsed() << " s\n";
  }
}



