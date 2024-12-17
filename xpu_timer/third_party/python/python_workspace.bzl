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
load("@xpu_timer_cfg//:xpu_config.bzl", "PYTHON_INCLUDE", "PYTHON_LIB")

def python_workspace():
    dynamic_local_repository(
        name = "python_lib",
        include = "PYTHON_INCLUDE",
        include_default_path = PYTHON_INCLUDE,
        lib = "PYTHON_LIB",
        lib_default_path = PYTHON_LIB,
        build_file = "//third_party/python:python.BUILD",
        template = True,
    )
