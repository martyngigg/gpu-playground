//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "HostTimer.h"
#include "Workspace2D.h"

#include <iostream>
#include <numeric>

namespace sandbox
{
  using std::adjacent_difference;
  using std::inner_product;

  Workspace2DPtr integrate_with_cpu(const Workspace2DPtr & workspace)
  {
    const size_t nVectors = workspace->nVectors;
    auto * result = new Workspace2D(nVectors, 1);
    std::cout << "Running integration on CPU...\n";
    HostTimer wallClock;
        
    #pragma omp parallel for
    for(size_t i = 0; i < nVectors; ++i)
    {
      const auto & yIn = workspace->dataY[i];
      const auto & xIn = workspace->dataX[i];
      DataArray widths(xIn.size() - 1);
      adjacent_difference(xIn.begin(), xIn.end(), widths.begin());
      result->dataY[i][0] = inner_product(yIn.begin(), yIn.end(), widths.begin(), 0.0);
    }

    std::cout << "Finished in " << wallClock.elapsed() << " s\n";
    return Workspace2DPtr(result);
  }
}



