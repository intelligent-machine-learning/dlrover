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

#include <gtest/gtest.h>

#include <cstdlib>  // for setenv and unsetenv

#include "tools/cpp/runfiles/runfiles.h"
#define private public
#include "xpu_timer/common/util.h"
#undef private

using bazel::tools::cpp::runfiles::Runfiles;

namespace xpu_timer {
namespace util {

class CommonEnvTest : public ::testing::Test {
 protected:
};

TEST_F(CommonEnvTest, TestEnv) {
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest());
  auto config_path = runfiles->Rlocation("config.ini");
  LOG(INFO) << config_path;
  std::ifstream file(config_path);
  setenv("XPU_TIMER_CONFIG", config_path.c_str(), 1);
  // test register value with default value
  REGISTER_ENV_VAR("POD_NAME", "test_123");
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<std::string>("POD_NAME"), "test_123");

  REGISTER_ENV_VAR("POD_NAME", 123);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<int>("POD_NAME"), 123);

  REGISTER_ENV_VAR("POD_NAME", true);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<bool>("POD_NAME"), true);

  // test register value with env var
  REGISTER_ENV_VAR("MY_INT_ENV_VARIABLE", -1);
  REGISTER_ENV_VAR("MY_BOOL_ENV_VARIABLE", true);
  REGISTER_ENV_VAR("MY_STR_ENV_VARIABLE", "undefine");
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<int>("MY_INT_ENV_VARIABLE"), 100);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<bool>("MY_BOOL_ENV_VARIABLE"), false);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<std::string>("MY_STR_ENV_VARIABLE"),
            "hello");

  // test non regieter value
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<int>("NODEFINE_INT_ENV_VARIABLE"),
            EnvVarRegistry::INT_DEFAULT_VALUE);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<bool>("NODEFINE_BOOL_ENV_VARIABLE"),
            EnvVarRegistry::BOOL_DEFAULT_VALUE);
  EXPECT_EQ(
      EnvVarRegistry::GetEnvVar<std::string>("NODEFINE_MY_STR_ENV_VARIABLE"),
      EnvVarRegistry::STRING_DEFAULT_VALUE);

  // test config file, bazel::tools::cpp::runfiles::Runfiles file can not use in
  // our source so we modify the ptree
  auto& pt = EnvVarRegistry::GetPtree();
  pt.put("XPU_TIMER_TIMELINE_TRACE_COUNT", 200);
  pt.put("XPU_TIMER_TIMELINE_PATH", "/root/test");
  REGISTER_ENV_VAR("XPU_TIMER_TIMELINE_TRACE_COUNT", -1);
  REGISTER_ENV_VAR("XPU_TIMER_TIMELINE_PATH", "test");
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_TIMELINE_TRACE_COUNT"),
            200);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_TIMELINE_PATH"),
            "/root/test");
  setenv("XPU_TIMER_TIMELINE_TRACE_COUNT", "666", 1);
  setenv("XPU_TIMER_TIMELINE_PATH", "FROM_ENV", 1);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<int>("XPU_TIMER_TIMELINE_TRACE_COUNT"),
            666);
  EXPECT_EQ(EnvVarRegistry::GetEnvVar<std::string>("XPU_TIMER_TIMELINE_PATH"),
            "FROM_ENV");
}

}  // namespace util
}  // namespace xpu_timer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
