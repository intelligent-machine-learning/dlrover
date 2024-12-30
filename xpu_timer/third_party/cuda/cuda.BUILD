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
    name = "cuda_nv",
    hdrs = glob(["include/nv/*"]),
    includes = ["include"],
)

cc_library(
    name = "cuda_nv_detail",
    hdrs = glob(["include/nv/detail/*"]),
    includes = ["include/nv/target"],
    deps = [":cuda_nv"],
)

cc_library(
    name = "cuda_headers",
    hdrs = glob(
        include=["include/**/*.h", "include/**/*.hpp"],
        exclude=["include/**/nccl*.h"]
    ),
    includes = ["include"],
    deps = [":cuda_nv", ":cuda_nv_detail"]
)
