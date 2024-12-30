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

#pragma once

#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/protos/hosting_service.pb.h"
#include "xpu_timer/server/hosting_service_server_client.h"

namespace xpu_timer {
using namespace util;
using namespace server;

class SignalHandler {
 public:
  SignalHandler(int signum);
  ~SignalHandler() { restoreOriginalHandler(); }

  static void customSignalHandler(int signum, siginfo_t *info, void *context);
  static void registerHandler(int signum);
  static void registerDefault(std::shared_ptr<ClientStub> stub);

 private:
  int signum_;
  bool is_hook_;
  struct sigaction original_action_;

  static inline std::unordered_map<int, std::unique_ptr<SignalHandler>>
      signal_handlers_;
  static inline std::unordered_map<int, std::string> signal_names_ = {
      {SIGABRT, "SIGABRT"}, {SIGBUS, "SIGBUS"},   {SIGFPE, "SIGFPE"},
      {SIGPIPE, "SIGPIPE"}, {SIGSEGV, "SIGSEGV"}, {SIGTERM, "SIGTERM"},
  };
  static inline std::shared_ptr<ClientStub> dump_stub_;

  void restoreOriginalHandler();
  void backtraceFrameInfo(SignalFrameRequest *request);
  void pushSignal();
};

}  // namespace xpu_timer
