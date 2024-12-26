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

# flake8: noqa: E501,E722,F841,E401
import os
import runpy
import sys
import time
from pathlib import Path

PY_EXCEPTION_FN = "xpu_timer_parse_python_exception"
CPP_EXCEPTION_FN = "xpu_timer_parse_cpp_exception"


def load_plugins(plugin_paths, pattern):
    fns = []
    for plugin in plugin_paths:
        try:
            plugin_namespace = runpy.run_path(plugin)
            if pattern in plugin_namespace:
                fns.append(plugin_namespace[pattern])
        except:
            print(f"[XPU_TIMER] error when load {plugin}", file=sys.stderr)
            # maybe file not found or import error
            continue
    return fns


def xpu_timer_parse_python_exception(exc_type, exc_value, exc_traceback):

    if exc_type is KeyboardInterrupt:
        return
    job_infos = {}
    job_infos["time"] = int(time.time())
    job_infos["pod_name"] = os.environ.get("POD_NAME", "UNKNOWN")
    job_infos["job_name"] = os.environ.get("ENV_ARGO_WORKFLOW_NAME", "UNKNOWN")
    job_infos["ip"] = os.environ.get("POD_IP", "UNKNOWN")
    job_infos["rank"] = int(os.environ.get("RANK", "-1"))

    plugin_paths = []
    path_dir = Path(__file__).parent
    plugin_from_env = os.environ.get("XPU_TIMER_EXIT_HOOK_PLUGIN", None)
    plugin_paths.append(path_dir / "dlrover_parse_exception.py")
    if plugin_from_env is not None:
        plugin_paths.extend(plugin_from_env.split(","))
    fns = load_plugins(plugin_paths, PY_EXCEPTION_FN)
    for fn, plugin in zip(fns, plugin_paths):
        try:
            fn(exc_type, exc_value, exc_traceback, job_infos)
        except:
            # ignore all exceptions
            print(f"[XPU_TIMER] error when running {plugin}", file=sys.stderr)
            continue


def xpu_timer_parse_cpp_exception(stack_infos):
    plugin_paths = []
    path_dir = Path(__file__).parent
    plugin_from_env = os.environ.get("XPU_TIMER_EXIT_HOOK_PLUGIN", None)
    plugin_paths.append(path_dir / "dlrover_parse_exception.py")
    if plugin_from_env is not None:
        plugin_paths.extend(plugin_from_env.split(","))
    fns = load_plugins(plugin_paths, CPP_EXCEPTION_FN)
    for fn, plugin in zip(fns, plugin_paths):
        try:
            fn(stack_infos)
        except:
            # ignore all exceptions
            print(f"[XPU_TIMER] error when running {plugin}", file=sys.stderr)
            continue
