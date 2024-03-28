#pragma once

namespace atorch {

class XpuTimer {
  /* This is interface for timing kernel using xpu event.
   */
 public:
  // return the duration of kernel execution in us.
  virtual uint64_t getDuration() = 0;
  // return the event is ready.
  virtual bool isReady() = 0;
  // return description of the kernel.
  virtual const std::string getName() = 0;
  // return type of the kernel.
  virtual const std::string getType() = 0;
  // return flop of the kernel.
  virtual const std::string getFlop() = 0;
  // start record event.
  virtual void startRecord() = 0;
  // end record event.
  virtual void endRecord() = 0;
  // do some prepare works, maybe resolve the symbol table or others.
  static void doPrepare(){};
};

}  // namespace atorch
