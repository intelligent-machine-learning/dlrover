#include "xpu_timer/common/platform.h"

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/util.h"

namespace xpu_timer {
namespace platform {
#if defined(XPU_NVIDIA)

std::string getDeviceName() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    XLOG(FATAL) << "No CUDA devices found, abort";
    std::abort();
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::string full_device_name;
  std::string device_name;

  try {
    // Tesla P100-PCIE-16GB
    // NVIDIA A100-SXM4-80GB
    full_device_name = util::split(deviceProp.name, " ").at(1);
    device_name = util::split(full_device_name, "-").at(0);
  } catch (const std::out_of_range &e) {
    XLOG(ERROR) << "Device name parsing error origin name is "
                << deviceProp.name << " Fall back to A100";
    device_name = "A100";
  }

  XLOG(INFO) << "Device name " << device_name << " origin " << deviceProp.name;
  return device_name;
}

#endif
}  // namespace platform
}  // namespace xpu_timer
