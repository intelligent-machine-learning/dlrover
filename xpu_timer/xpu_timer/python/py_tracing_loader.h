#pragma once

#include <string>
#include <vector>

#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/python/py_tracing_data.h"

namespace xpu_timer {
namespace py_tracing_manager {

class PyTracingLibrary : public LibraryLoader {
 public:
  PyTracingLibrary(const std::string&);
  using XpuTimerRegisterTracingFunc = void (*)(const char**, int, char**);
  using GetFullTracingDataArrayFunc = XpuTimerPyTracingDataArray* (*)(int);
  using GetPartialTracingDataArrayFunc = XpuTimerPyTracingDataArray* (*)(int);
  using ReturnTracingDataArrayFunc = void (*)(XpuTimerPyTracingDataArray*, int,
                                              int);
  using SwitchTracingFunc = void (*)(int);
  using GetTracingCountFunc = int64_t (*)(int);

  std::vector<std::string> Register(const std::vector<std::string>& names);
  XpuTimerPyTracingDataArray* GetFullTracingData(int);
  XpuTimerPyTracingDataArray* GetPartialTracingData(int);
  void ReturnTracingData(XpuTimerPyTracingDataArray* data, int type, int name);
  void SwitchTracing(int flag);
  int64_t GetTracingCount(int name);

 private:
  XpuTimerRegisterTracingFunc register_tracing_;
  GetFullTracingDataArrayFunc get_tracing_data_;
  GetPartialTracingDataArrayFunc get_partial_tracing_data_;
  ReturnTracingDataArrayFunc return_tracing_data_;
  SwitchTracingFunc switch_tracing_;
  GetTracingCountFunc get_tracing_count_;
};

}  // namespace py_tracing_manager
}  // namespace xpu_timer
