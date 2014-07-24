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

  void integrate_on_cpu(const size_t nvec, const size_t npts)
  {
    std::cout << "Running integration on CPU...\n";
    HostTimer wallClock;

    auto workspace = createCosineWorkspace(nvec, npts);

    auto & dataY0 = workspace->dataY[0];
    auto & dataX0 = workspace->dataX[0];

    DataArray widths(dataX0.size() - 1);
    std::adjacent_difference(dataX0.begin(), dataX0.end(), widths.begin());
    DataType sumY = std::inner_product(dataY0.begin(), dataY0.end(), widths.begin(), 0.0);
    std::cout << "Integral = " << sumY << "\n";

    std::cout << "Finished in " << wallClock.elapsed() << " s\n";
  }
}



