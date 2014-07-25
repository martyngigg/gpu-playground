//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include "Workspace2D.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace sandbox
{
  namespace
  {
    // Max value of X axis
    DataType XMAX = 0.5*M_PI;
  }

  /**
   * 2D "array" with nvecs sets of 1D vectors of size npts for Y & X
   * @param nvec Number of vectors
   * @param nvec Number of pts in each vector
   */
  Workspace2D::Workspace2D(const size_t nvecs, const size_t npts) :
    dataY(nvecs, DataArray(npts, DataType())),
    dataX(nvecs, DataArray(npts, DataType())), nVectors(nvecs), 
    nPts(npts)
  {
  }

  //------------------------------------------------------------------------------
  // Creation helpers
  //------------------------------------------------------------------------------

  namespace
  {
    struct Linspace
    {
      Linspace(DataType end, size_t n)
      {
        current = DataType();
        delta = (end - current)/n;
      }

      DataType operator()()
      {
        current += delta;
        return current;
      }

      DataType delta;
      DataType current;
    };
  }

  /**
   * Create a 2D workspace where x pts are 2pi/npts & Y are cos(x)
   * @param nvec Number of vectors
   * @param nvec Number of pts in each vector
   */
  Workspace2DPtr createCosineWorkspace(const size_t nvec,
                                              const size_t npts)
  {
    using std::cos;
    auto *typedCos = (DataType (*)(DataType))cos;

    auto wksp = Workspace2DPtr(new Workspace2D(nvec, npts));
    for(size_t i = 0; i < nvec; ++i)
    {
      auto & dataX = wksp->dataX[i];
      std::generate(dataX.begin(), dataX.end(), Linspace(XMAX, npts));
      auto & dataY = wksp->dataY[i];
      std::transform(dataX.begin(), dataX.end(), dataY.begin(), typedCos);
    }

    return wksp;
  }

}
