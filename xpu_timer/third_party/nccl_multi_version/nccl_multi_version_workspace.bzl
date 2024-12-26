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
load("//:workspace.bzl", "nccl_code_repository")

def nccl_multi_version_workspace():
    nccl_code_repository(
        name = "nccl_multi_version",
        urls = [
            "https://github.com/NVIDIA/nccl/archive/refs/tags/v2.18.5-1.zip",
            "https://github.com/NVIDIA/nccl/archive/refs/tags/v2.20.5-1.zip",
            "https://github.com/NVIDIA/nccl/archive/refs/tags/v2.21.5-1.zip",
        ],
        sha256s = [
            "900c76df10c35d25d09c56772a4b56b32f3bac9267f62bd9b32eaf0e0304a7d9",
            "619b960b793aa41f946b6177be3752ee0005a04028550124b8e8e862da994598",
            "7c1580dcd4756f031961a28e75384be8ea50d328dcc61b5bd46b29ea82fad0f4",
        ],
        # same as in runtime's VERSION
        # get from `strings ${NCCL_LIB_PATH} | grep -E "version.*cuda" | awk -F"[ +]" '{print $1 "_" $3}'`
        version_tags = ["NCCL_2.18.5", "NCCL_2.20.5", "NCCL_2.21.5"],
        build_file = "//third_party/nccl_multi_version:nccl.BUILD",
        src_template_file = "//third_party/nccl_multi_version:template.nccl_parser.cc",
        version_script_file = "//third_party/nccl_multi_version:template.nccl_parser_function.lds",
    )
