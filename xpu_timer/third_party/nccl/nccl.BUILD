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

genrule(
    name = "nccl_h",
    srcs = glob(["**"]),
    outs = ["src/include/nccl.h"],
    cmd = "(cd $$(dirname $(location src/Makefile)) && BUILDDIR=`pwd` make `pwd`/include/nccl.h && cd - && cp $$(dirname $(location src/Makefile))/include/nccl.h $(@D)/nccl.h)"
)

filegroup(
    name = "nccl_source",
    srcs = glob(["src/include/**/*.h", "src/include/**/*.hpp"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl_headers",
    hdrs = [":nccl_h", ":nccl_source"],
    deps = [
        "@cuda//:cuda_headers",
    ],
    linkopts = ["-ldl"],
    includes = ["src/include"],
)
