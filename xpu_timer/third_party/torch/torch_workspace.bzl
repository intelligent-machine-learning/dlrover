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

load("//:workspace.bzl", "dynamic_local_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@xpu_timer_cfg//:xpu_config.bzl", "TORCH_PATH")

def torch_workspace():
    dynamic_local_repository(
        name = "torch",
        include_default_path = TORCH_PATH,
        build_file = "//third_party/torch:torch.BUILD",
        include = "TORCH_HOME",
    )

    http_archive(
        name = "flash_attn",
        build_file = "//third_party/torch:flash_attn.BUILD",
        strip_prefix = "flash-attention-fa2_pack_glm_mask",
        urls = ["https://github.com/intelligent-machine-learning/flash-attention/archive/fa2_pack_glm_mask.tar.gz"],
        sha256 = "1e2ab9fb7198c57f7f332e0b30f988bb47fb66220af56de6411d8744805e2e2b",
    )
