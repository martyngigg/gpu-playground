// -*- mode: c++; -*-
//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include "CUDATimer.h"

#include <ostream>
#include <sstream>
#include <string>

namespace sandbox
{

  /** Constructor.
   *  Instantiating the object starts the timer.
   */
  CUDATimer::CUDATimer()
  {
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);

    cudaEventRecord(m_start);
  }

  /// Destructor
  CUDATimer::~CUDATimer()
  {
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
  }

  /** Returns the wall-clock time elapsed in seconds since the Timer object's creation, or the last call to elapsed
   *
   * @param reset :: set to true to reset the clock (default)
   * @return time in seconds
   */
  float CUDATimer::elapsed(bool reset)
  {
    float elapsed = this->elapsed_no_reset();
    if(reset) this->reset();
    return elapsed;
  }

  /** Returns the wall-clock time elapsed in seconds since the Timer object's creation, or the last call to elapsed
   *
   * @return time in seconds
   */
  float CUDATimer::elapsed_no_reset() const
  {
    cudaEventRecord(m_stop);
    cudaEventSynchronize(m_stop);
    float milli(0.0f);
    cudaEventElapsedTime(&milli, m_start, m_stop);
    return milli/1000.0f;
  }

  /// Explicitly reset the timer.
  void CUDATimer::reset()
  {
    cudaEventRecord(m_start);
  }

  /// Convert the elapsed time (without reseting) to a string.
  std::string CUDATimer::str() const
  {
    std::stringstream buffer;
    buffer << this->elapsed_no_reset() << "s";
    return buffer.str();
  }

  /// Convenience function to provide for easier debug printing.
  std::ostream& operator<<(std::ostream& out, const CUDATimer& obj)
  {
    out << obj.str();
    return out;
  }

} // namespace sandbox




