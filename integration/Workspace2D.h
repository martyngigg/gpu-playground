#ifndef WORKSPACE2D_H_
#define WORKSPACE2D_H_
//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include <memory>
#include <vector>

namespace sandbox
{
  //------------------------------------------------------------------------------
  // Typedefs
  //------------------------------------------------------------------------------
#ifdef DBL_DATA
  /// Datatype
  typedef double DataType;
#else
  /// Datatype
  typedef float DataType;
#endif
  
  /// Array of data
  typedef std::vector<DataType> DataArray;
  
  //------------------------------------------------------------------------------
  // Workspace2D
  //------------------------------------------------------------------------------
  
  /**
   * Holds X & Y information in a 2D grid
   */
  struct Workspace2D
  {
    /**
     * 2D "array" with nvecs sets of 1D vectors of size npts for Y & X
     */
    Workspace2D(const size_t nvecs, const size_t npts);

    /// Signal values
    std::vector<DataArray> dataY;
    /// X values
    std::vector<DataArray> dataX;
    /// Number of 1D vectors
    size_t nVectors;
    /// Number of pts per vector
    size_t nPts;
  };
  
  typedef std::auto_ptr<Workspace2D> Workspace2DPtr;

  //------------------------------------------------------------------------------
  // Creation helpers
  //------------------------------------------------------------------------------
  /// Create a set of cosine vectors
  Workspace2DPtr createCosineWorkspace(const size_t nvec, const size_t npts);
  
}

#endif /* WORKSPACE2D_H_ */
