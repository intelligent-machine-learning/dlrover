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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@xpu_timer_cfg//:xpu_config.bzl", "NCCL_VERSION")

def nccl_workspace():
    # only for nccl 2.18.5, if you change version, you should generate the headers of `devcomm`
    http_archive(
        name = "nccl",
        build_file = "//third_party/nccl:nccl.BUILD",
        strip_prefix = "nccl-{}".format(NCCL_VERSION),
        urls = ["https://github.com/NVIDIA/nccl/archive/refs/tags/v{}.tar.gz".format(NCCL_VERSION)],
    )
