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

#include "xpu_timer/common/logging.h"

namespace xpu_timer {

void setLoggingPath(bool is_brpc_server) {
  std::string logging_path =
      util::EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_LOGGING_DIR");
  if (logging_path != util::EnvVarRegistry::STRING_DEFAULT_VALUE)
    util::ensureDirExists(logging_path);
  // brpc server must logging to file
  if (logging_path == "stdout" && !is_brpc_server) return;
  std::string rank_str = std::to_string(util::config::GlobalConfig::rank);
  std::string log_file_name =
      is_brpc_server ? "/xpu_timer_daemon.log" : "/xpu_hook.log." + rank_str;

  if (logging_path != util::EnvVarRegistry::STRING_DEFAULT_VALUE) {
    logging_path =
        logging_path + (is_brpc_server ? "/xpu_timer_daemon.log"
                                       : "/xpu_hook.log." + rank_str);
  } else {
    if (is_brpc_server)
      logging_path = "/tmp/xpu_timer_daemon.log";
    else
      logging_path = "/tmp/xpu_hook.log." + rank_str;
  }
  logging::LoggingSettings log_settings;
  log_settings.logging_dest = logging::LOG_TO_FILE;
  std::cout << "XPU_TIMER logging to " << logging_path << std::endl;
  log_settings.log_file = logging_path.c_str();

  REGISTER_ENV_VAR("XPU_TIMER_LOGGING_DIR",
                   util::EnvVarRegistry::STRING_DEFAULT_VALUE);
  if (util::EnvVarRegistry::GetEnvVar<bool>("XPU_TIMER_LOGGING_APPEND"))
    log_settings.delete_old = logging::APPEND_TO_OLD_LOG_FILE;
  else
    log_settings.delete_old = logging::DELETE_OLD_LOG_FILE;
  logging::InitLogging(log_settings);
}
}  // namespace xpu_timer
