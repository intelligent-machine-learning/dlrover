# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
build_time=$(awk -F'"' '/STABLE_BUILD_TIME/{print $2}' bazel-out/stable-status.txt)
git_version=$(awk '/STABLE_GIT_COMMIT/{print $2}' bazel-out/stable-status.txt)
build_type=$(awk '/STABLE_BUILD_TYPE/{print $2}' bazel-out/stable-status.txt)
build_platform=$(awk '/\<STABLE_BUILD_PLATFORM\>/{print $2}' bazel-out/stable-status.txt)
build_platform_version=$(awk '/STABLE_BUILD_PLATFORM_VERSION/{print $2}' bazel-out/stable-status.txt)
echo $build_time
cat <<EOF >$1
// this file is auto generated, do not modify!
#pragma once

namespace xpu_timer {
namespace util {
static constexpr const char* git_version = "$git_version";
static constexpr const char* build_time = "$build_time";
static constexpr const char* build_type = "$build_type";
static constexpr const char* build_platform = "$build_platform";
static constexpr const char* build_platform_version = "$build_platform_version";
}  // namespace util
}  // namespace xpu_timer
EOF
