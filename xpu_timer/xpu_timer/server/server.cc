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

#include <butil/logging.h>
#include <bvar/bvar.h>
#include <gflags/gflags.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/util.h"
#include "xpu_timer/common/version.h"
#include "xpu_timer/server/hosting_service_server_client.h"
#include "xpu_timer/server/python_plugin.h"

DEFINE_string(server_path, "/tmp/xpu_timer_daemon.sock", "server path");
DEFINE_string(log_path, "/tmp/xpu_timer_daemon.log", "log path");
DEFINE_int32(local_world_size, 0, "local world size");
DEFINE_int32(prometheus_port, 18889, "port of prometheus service");
DEFINE_int32(try_start, 0, "try start, for check runtime env");

int main(int argc, char* argv[]) {
  GFLAGS_NS::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_try_start) {
    LOG(INFO) << "xpu_timer_daemon test ok";
    return 0;
  }

  xpu_timer::util::REGISTER_ENV();
  xpu_timer::util::config::setUpGlobalConfig();
  xpu_timer::setLoggingPath(true);

  // logging::LoggingSettings log_settings;
  // log_settings.logging_dest = logging::LOG_TO_FILE;
  // log_settings.log_file = FLAGS_log_path.c_str();
  // log_settings.delete_old = logging::DELETE_OLD_LOG_FILE;

  // logging::InitLogging(log_settings);
  {
    pid_t pid = getpid();
    std::ofstream out_file("/tmp/xpu_timer_daemon.pid");
    if (!out_file) {
      LOG(ERROR) << "Error opening file for writing";
      return 0;  // TODO start server maybe
    }
    out_file << pid << std::endl;
  }
  LOG(INFO) << "xpu_timer git version is " << xpu_timer::util::git_version;
  LOG(INFO) << "xpu_timer build time is " << xpu_timer::util::build_time;
  LOG(INFO) << "[ENV] xpu_timer build type is " << xpu_timer::util::build_type;
  LOG(INFO) << "Daemon run at " << FLAGS_server_path;

  ::bvar::xpu_timer::bvar_enable_sampling_from_xpu_timer = true;

  xpu_timer::server::MainServer mans(FLAGS_server_path, FLAGS_local_world_size,
                                     FLAGS_prometheus_port,
                                     FLAGS_local_world_size);
  mans.start(xpu_timer::server::MainServer::LOCAL_RANK_0_SERVER);
  LOG(INFO) << "Daemon run into barrier start_work_barrier";
  xpu_timer::util::InterProcessBarrier(
      FLAGS_local_world_size + 1, FLAGS_local_world_size, "start_work_barrier");

  xpu_timer::server::xpuTimerDaemonSetUpPython();
  mans.join();
  return 0;
}
