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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

def perfetto_workspace():
    http_archive(
        name = "com_google_perfetto",
        urls = ["https://github.com/google/perfetto/archive/refs/tags/v45.0.tar.gz"],
        strip_prefix = "perfetto-45.0",
        build_file = "//third_party/perfetto:perfetto.BUILD",
        sha256 = "dcb815fb54370fa20a657552288016cb66e7a98237c1a1d47e7645a4325ac75e"
    )
