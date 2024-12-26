// Copyright 2024 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xpu_timer/common/signal_handler.h"

#include <cxxabi.h>
#include <execinfo.h>
#include <libunwind.h>

#include <chrono>
#include <ctime>

namespace xpu_timer {
using namespace util;
using namespace server;

void SignalHandler::registerDefault(std::shared_ptr<ClientStub> stub) {
  SignalHandler::registerHandler(SIGABRT);
  SignalHandler::registerHandler(SIGBUS);
  SignalHandler::registerHandler(SIGFPE);
  SignalHandler::registerHandler(SIGPIPE);
  SignalHandler::registerHandler(SIGSEGV);
  SignalHandler::registerHandler(SIGTERM);
  dump_stub_ = stub;
}

SignalHandler::SignalHandler(int signum) : signum_(signum) {
  struct sigaction sa {};
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = &SignalHandler::customSignalHandler;
  sigemptyset(&sa.sa_mask);

  if (sigaction(signum_, &sa, &original_action_) == -1) {
    XLOG(INFO) << "Failed to set custom signal handler for " << signum;
    is_hook_ = false;
  }
  XLOG(INFO) << "Registering custom signal handler for " << signum;
  is_hook_ = true;
}

void SignalHandler::registerHandler(int signum) {
  if (signal_handlers_.find(signum) == signal_handlers_.end()) {
    signal_handlers_[signum] = std::make_unique<SignalHandler>(signum);
    return;
  }
  XLOG(INFO) << "Signal " << signum << " has register already";
}

void SignalHandler::customSignalHandler(int signum, siginfo_t *info,
                                        void *context) {
  auto it = signal_handlers_.find(signum);
  if (it != signal_handlers_.end()) {
    XLOG(INFO) << "Caught " << signal_names_[signum] << " (signal " << signum
               << ") at address " << info->si_addr;
    it->second->pushSignal();
    it->second->restoreOriginalHandler();
  }
  raise(signum);
}

void SignalHandler::restoreOriginalHandler() {
  if (is_hook_) {
    if (sigaction(signum_, &original_action_, nullptr) == -1)
      XLOG(ERROR) << "Failed to restore original signal handler for "
                  << signum_;
    else
      is_hook_ = false;
  }
}
void SignalHandler::backtraceFrameInfo(SignalFrameRequest *request) {
  unw_cursor_t cursor;
  unw_context_t context;

  unw_getcontext(&context);
  unw_init_local(&cursor, &context);

  XLOG(INFO)
      << "[XPU_TIMER_SIGNAL] Stack trace with function names and offsets:";
  while (unw_step(&cursor) > 0) {
    SignalFrameRequest::FrameInfo *frame = request->add_frame_infos();
    unw_word_t offset, pc;
    char func_name[8192];

    if (unw_get_reg(&cursor, UNW_REG_IP, &pc) != 0) {
      break;
    }
    frame->set_address(pc);
    frame->set_offset(offset);

    if (unw_get_proc_name(&cursor, func_name, sizeof(func_name), &offset) ==
        0) {
      int status;
      char *demangled_name =
          abi::__cxa_demangle(func_name, nullptr, nullptr, &status);
      const char *function_name =
          (status == 0 && demangled_name) ? demangled_name : func_name;

      XLOG(INFO) << "[XPU_TIMER_SIGNAL] 0x" << std::hex << pc << ": ("
                 << function_name << "+0x" << std::hex << offset << ")";
      frame->set_function_name(function_name);
      free(demangled_name);  // Free the demangled name if it was used
    } else {
      frame->set_function_name("??");
      XLOG(INFO) << "[XPU_TIMER_SIGNAL] 0x" << std::hex << pc
                 << ": -- error: unable to obtain symbol name";
    }
  }
}

void SignalHandler::pushSignal() {
  SignalFrameRequest request;
  request.set_signal(signum_);
  request.set_rank(config::GlobalConfig::local_rank);
  request.set_pod_name(config::GlobalConfig::pod_name);
  request.set_job_name(config::GlobalConfig::job_name);
  uint64_t utc_timestamp = static_cast<uint64_t>(
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  request.set_timestamp(utc_timestamp);
  backtraceFrameInfo(&request);
  dump_stub_->pushSignalFrameInfo(&request);
  XLOG(INFO) << "rank " << request.rank() << " push ok";
}

}  // namespace xpu_timer
