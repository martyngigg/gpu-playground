#ifndef CUDATIMER_H_
#define CUDATIMER_H_

#include <cuda.h>
#include <string>

namespace sandbox
{
  class CUDATimer
  {
  public:
    CUDATimer();
    virtual ~CUDATimer();

    float elapsed(bool reset = true);
    float elapsed_no_reset() const;
    std::string str() const;
    void reset();

  private:
    cudaEvent_t m_start, m_stop;
  };

}
#endif /* CUDATIMER_H_ */
