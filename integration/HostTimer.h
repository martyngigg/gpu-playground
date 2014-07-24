#ifndef TIMER_H_
#define TIMER_H_

#include <iosfwd>

#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

namespace sandbox
{
  class HostTimer
  {
  public:
    HostTimer();
    virtual ~HostTimer();

    float elapsed(bool reset = true);
    float elapsed_no_reset() const;
    std::string str() const;
    void reset();

  private:
    // The type of this variable is different depending on the platform
  #ifdef _WIN32
    clock_t
  #else
    timeval
  #endif
    m_start;   ///< The starting time (implementation dependent format)
  };

}
#endif /* TIMER_H_ */
