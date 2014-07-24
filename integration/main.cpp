//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include <cstddef>

using std::size_t;

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
namespace sandbox
{
  void integrate_on_cpu(const size_t, const size_t);
  void integrate_with_cuda(const size_t, const size_t);
}

int main()
{
  using namespace sandbox;

  const size_t nvec(10000), npts(10000);
  integrate_on_cpu(nvec, npts);
  integrate_with_cuda(nvec, npts);
}




