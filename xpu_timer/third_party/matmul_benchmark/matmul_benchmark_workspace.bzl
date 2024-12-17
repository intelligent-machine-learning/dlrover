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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

def matmul_benchmark_workspace():

    http_file(
      name = "cublaslt_gemm_bin",
      sha256 = "3f81a13c63724ab32dbefa4a723f4bdeeaf11da9bb19be7b28b2116d96e7f221",
      url = "https://dlrover.oss-cn-beijing.aliyuncs.com/atorch/libs/cublaslt_gemm",
      executable = True,
    )
    
    http_file(
      name = "cublas_benchmark_bin",
      sha256 = "a64baee971a41fd3cd2900305af4fad33ba50e4e9719aa86e59670e8c7ccd002",
      url = "https://dlrover.oss-cn-beijing.aliyuncs.com/atorch/libs/cublas_benchmark",
      executable = True,
    )
