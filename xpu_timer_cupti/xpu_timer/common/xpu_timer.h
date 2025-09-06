#pragma once
#include <butil/logging.h>

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "xpu_timer/common/constant.h"
#include "xpu_timer/common/util.h"
namespace xpu_timer {

namespace config = xpu_timer::util::config;
namespace util = xpu_timer::util;
namespace constant = xpu_timer::constant;
using Labels = std::map<std::string, std::string>;

struct PyStackTrace {
  std::string filename;
  std::string function_name;
  int line;
};

using PyStack = std::vector<PyStackTrace>;

class XpuTimer {
  /* This is interface for timing kernel using xpu event.
   */
 public:
  // lifetime of py_stack is managed in ::xpu_timer::GpuTimerManager::doWork
  PyStack* py_stack{nullptr};
  // this rebuild the XpuTimer object, typically, this function is called on
  // background thread, not to waste cycles in thread which launching kernel.
  virtual void reBuild() = 0;
  // return the code of tracing kernel
  virtual int getTraceCode() = 0;
  // return the id of tracing event
  virtual uint64_t getTraceId() = 0;
  // return the start timestamp of kernel
  virtual time_t getExecuteTimeStamp() = 0;
  // return the launch timestamp of kernel
  virtual time_t getLaunchTimeStamp() = 0;
  // return the duration of kernel execution in us.
  virtual uint64_t getDuration() = 0;
  // return the event is ready.
  virtual bool isReady() = 0;
  // return the event is skipped.
  virtual bool isSkipped();
  // return description of the kernel. This is key of `BvarPrometheus` objects
  virtual const std::string getName() = 0;
  // return type of the kernel. the type should be static string_view in global
  // scope.
  virtual const std::string_view& getType() = 0;
  // return problem size of the kernel.
  virtual const uint64_t getProblemSize() = 0;
  // start record event.
  virtual void startRecord() = 0;
  // end record event.
  virtual void endRecord() = 0;
  // get extra label message, the label is used in prometheus
  virtual Labels getExtraLabels() = 0;
  // check the event is hang or not, if hang, we can dump all stack.
  virtual bool isHang(time_t timeout) = 0;
  // ignore hang detect for this event, only set for allreduce in uint8
  // this is torch.dist.barrier()
  virtual bool ignoreHang() = 0;
  // host side code
  virtual bool isHost() = 0;
  // judge the timer should be traced
  virtual bool isValidateToTrace() = 0;

  /*
   * ===================================
   * Static interface
   * ===================================
   * The following member functions are
   * MUST implement in subclass as static
   * ===================================
   */
  // do some prepare works, maybe resolve the symbol table or others.
  static void doPrepare();

  // do some destroy works, maybe release hooks from CuptiActivity.
  static void doDestroy();

  // dump the trace meta, mapping from trace_code -> kernel_name
  static void dumpTraceMeta(const std::string& path,
                            const std::vector<std::string>& extra);

  // reset timer for each stream, it's use for dumping trace, to get real
  // timestamp when kernel launched on this stream
  static void doPrepareForDumpTrace();

  // get bucket level, depends on different platform with different dtype
  static int getBucket(double performance, const std::string& dtype);
};

class LibraryLoader {
 protected:
  void* handle_;
  bool can_use_;
  const std::string library_path_;
  void LoadLibrary();

 public:
  LibraryLoader(const std::string& library_path);
};

}  // namespace xpu_timer
