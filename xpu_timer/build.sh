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

set -exuo pipefail

for arg in "$@"; do
  if [[ "$arg" == "-h" ]] || [[ "$arg" == "--help" ]]; then
    python3 config.py "$@"
    exit 0
  fi
done
BUILD_IGNORE_GIT=$(git rev-parse --is-inside-work-tree &>/dev/null && echo 0 || echo 1) python3 config.py "$@"

pkill -9 -f xpu_timer_daemon || true

# test env
bazelisk test //test/... || exit 1

platform=$(cat .build_platform)
platform_version=$(cat .platform_version)

# build python package
bazelisk build //py_xpu_timer/...

awk '/STABLE_GIT_COMMIT/{print "__version__ = \""$2"\""}' bazel-out/stable-status.txt >bazel-bin/py_xpu_timer/py_xpu_timer/version.py
build_time=$(awk -F'"' '/STABLE_BUILD_TIME/{print $2}' bazel-out/stable-status.txt)
build_type=$(awk '/STABLE_BUILD_TYPE/{print $2}' bazel-out/stable-status.txt)
build_platform=$(awk '/\<STABLE_BUILD_PLATFORM\>/{print $2}' bazel-out/stable-status.txt)
echo "__build_time__ = \"$build_time\"" >>bazel-bin/py_xpu_timer/py_xpu_timer/version.py
echo "__build_type__ = \"$build_type\"" >>bazel-bin/py_xpu_timer/py_xpu_timer/version.py
echo "__build_platform__ = \"$build_platform\"" >>bazel-bin/py_xpu_timer/py_xpu_timer/version.py
echo "__build_platform_version__ = \"$platform_version\"" >>bazel-bin/py_xpu_timer/py_xpu_timer/version.py
echo "__all__ = ['__build_time__', '__version__', '__build_type__', '__build_platform__', '__build_platform_version__']" >>bazel-bin/py_xpu_timer/py_xpu_timer/version.py


root=`pwd`

mkdir dist_bin || true
rm -rf dist_bin/*

cd bazel-bin/py_xpu_timer
export XPU_PLATFORM=$platform_version
python3 -m build || python3 setup.py bdist_wheel
export LITE_MODE=1
python3 -m build || python3 setup.py bdist_wheel

cp -f dist/* $root/dist_bin
