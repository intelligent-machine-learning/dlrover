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
