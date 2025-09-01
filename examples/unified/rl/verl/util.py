# Copyright 2025 The DLRover Authors. All rights reserved.
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

import threading
from concurrent.futures import Future
from functools import wraps
from typing import cast

import ray
from verl.single_controller.base import Worker as VerlWorker
from verl.single_controller.base.decorator import MAGIC_ATTR
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.ray.base import func_generator

from dlrover.python.unified.api.runtime.rpc_helper import (
    RoleGroup,
    export_rpc_method,
    rpc,
)


def export_verl_worker_rpc(core):
    """Export all RPC methods with veRL metadata"""
    for f in dir(core):
        v = getattr(core, f)
        if callable(v) and hasattr(v, MAGIC_ATTR):
            export_rpc_method(f, v)


class BaseWorker:
    def __init__(self, core: VerlWorker) -> None:
        self.end = threading.Event()
        export_verl_worker_rpc(core)

    def run(self):
        self.end.wait()

    @rpc()
    def job_end(self):
        """For trainer call"""
        self.end.set()


def notify_job_end(*roles: str):
    futures = [
        RoleGroup(role, optional=True).call("job_end") for role in roles
    ]
    [f.result() for f in futures]


class MyWorkerGroup(RoleGroup, WorkerGroup):
    """WorkerGroup based on DLRover's RoleGroup"""

    def __init__(self, role: str, worker_cls: type) -> None:
        RoleGroup.__init__(self, role)
        WorkerGroup._bind_worker_method(
            cast(WorkerGroup, self), worker_cls, func_generator
        )
        self._patch_ray_get()

    @property
    def world_size(self) -> int:
        return len(self.actors)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.call(method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.call_rank0(method_name, *args, **kwargs)

    # Let linters know this class is dynamic
    def __getattr__(self, item):
        return super().__getattribute__(item)

    @staticmethod
    def _patch_ray_get():
        """execute_all returns Future instead ObjectRef, patch to support ray.get"""
        raw_ray_get = ray.get
        if hasattr(raw_ray_get, "_patched_for_future"):
            return

        @wraps(raw_ray_get)
        def wrap_ray_get(obj, *args, **kwargs):
            if isinstance(obj, Future):
                return obj.result()
            return raw_ray_get(obj, *args, **kwargs)

        setattr(wrap_ray_get, "_patched_for_future", True)
        ray.get = wrap_ray_get
