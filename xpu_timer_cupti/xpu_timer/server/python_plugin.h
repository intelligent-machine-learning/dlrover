#pragma once

#include "xpu_timer/protos/hosting_service.pb.h"

namespace xpu_timer {
namespace server {

void xpuTimerDaemonSetUpPython();
void runPythonPlugin(const SignalFrameRequest* request);
}  // namespace server
}  // namespace xpu_timer
