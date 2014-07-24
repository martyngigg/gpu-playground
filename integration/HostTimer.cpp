//------------------------------------------------------------------------------
// Includes
//------------------------------------------------------------------------------
#include "HostTimer.h"

#include <ostream>
#include <sstream>
#include <string>

namespace sandbox
{

  /** Constructor.
   *  Instantiating the object starts the timer.
   */
  HostTimer::HostTimer()
  {
  #ifdef _WIN32
    m_start = clock();
  #else /* linux & mac */
    gettimeofday(&m_start,0);
  #endif
  }

  /// Destructor
  HostTimer::~HostTimer()
  {}

  /** Returns the wall-clock time elapsed in seconds since the Timer object's creation, or the last call to elapsed
   *
   * @param reset :: set to true to reset the clock (default)
   * @return time in seconds
   */
  float HostTimer::elapsed(bool reset)
  {
    float retval = elapsed_no_reset();
    if (reset) this->reset();
    return retval;
  }

  /** Returns the wall-clock time elapsed in seconds since the Timer object's creation, or the last call to elapsed
   *
   * @return time in seconds
   */
  float HostTimer::elapsed_no_reset() const
  {
  #ifdef _WIN32
    clock_t now = clock();
    const float retval = float(now - m_start)/CLOCKS_PER_SEC;
  #else /* linux & mac */
    timeval now;
    gettimeofday(&now,0);
    const float retval = float(now.tv_sec - m_start.tv_sec) +
        float(static_cast<float>(now.tv_usec - m_start.tv_usec)/1e6);
  #endif
    return retval;
  }

  /// Explicitly reset the timer.
  void HostTimer::reset()
  {
  #ifdef _WIN32
    m_start = clock();
  #else /* linux & mac */
    timeval now;
    gettimeofday(&now,0);
    m_start = now;
  #endif
  }

  /// Convert the elapsed time (without reseting) to a string.
  std::string HostTimer::str() const
  {
    std::stringstream buffer;
    buffer << this->elapsed_no_reset() << "s";
    return buffer.str();
  }

  /// Convenience function to provide for easier debug printing.
  std::ostream& operator<<(std::ostream& out, const HostTimer& obj)
  {
    out << obj.str();
    return out;
  }

} // namespace sandbox




