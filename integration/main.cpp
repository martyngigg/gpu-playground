//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include <cstddef>
#include <iostream>

#include "Workspace2D.h"

using std::size_t;
using namespace sandbox;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
namespace sandbox
{
  Workspace2DPtr integrate_with_cpu(const Workspace2DPtr & workspace);
  Workspace2DPtr integrate_with_cuda(const Workspace2DPtr & workspace);
  Workspace2DPtr integrate_with_thrust(const Workspace2DPtr & workspace);
}

int main()
{
  const size_t nvec(100000), npts(10000);
  std::cout << "Start timing tests with nvec=" << nvec << ", npts=" << npts << "\n";
  auto input = createCosineWorkspace(nvec, npts);

  integrate_with_cpu(input);
  integrate_with_thrust(input);
  integrate_with_cuda(input);
}




