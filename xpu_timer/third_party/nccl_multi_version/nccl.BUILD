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

package(
    default_visibility=["//visibility:public"]
)

cc_library(
    name = "nccl_headers",
    srcs = glob(["nccl/*/src/include/**/*.h", "nccl/*/src/include/**/*.hpp",
                 "nccl/*/*.h", "nccl/*/*.hpp"]),
    deps = [
        "@cuda//:cuda_headers",
    ],
    linkopts = ["-ldl"],
    includes = ["nccl"],
)

cc_library(
    name = "nccl_h",
    hdrs = ["nccl.h"],
    deps = [
        "@cuda//:cuda_headers",
    ],
    includes = ["."]
    
)


cc_library(
    name = "nccl_parser",
    srcs = glob(["nccl*.cc"]),
    copts = [
        "-fPIC",
    ],
    linkopts = [
        "-shared",
        "-Wl,--version-script=$(location :nccl.lds)",
        "-fuse-ld=bfd",
        "-L/usr/local/cuda/lib64",
        "-lcudart",
    ],
    deps = [":nccl.lds", "@cuda//:cuda_headers", ":nccl_headers"],
    # force to keep all syms, if not, linker will remove useless symbols.
    alwayslink = True,
)
