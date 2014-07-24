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
	};

  //------------------------------------------------------------------------------
  // Creation helpers
  //------------------------------------------------------------------------------
	/// Create a set of cosine vectors
	std::auto_ptr<Workspace2D> createCosineWorkspace(const size_t nvec,
	                                                 const size_t npts);

}

#endif /* WORKSPACE2D_H_ */
