# Copyright 2023 The DLRover Authors. All rights reserved.
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

import os

import ray

from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.stats.stats_backend import LocalFileStateBackend
from dlrover.python.scheduler.job import ElasticJob, JobArgs, NodeArgs
from dlrover.python.scheduler.kubernetes import (
    convert_cpu_to_decimal,
    convert_memory_to_mb,
)


def parse_bool(s: str):
    return s.lower() in ["true", "yes", "t", "y"]


@ray.remote
class RayWorker:  # pragma: no cover
    def __init__(self):
        pass

    def exec_module(self):
        pass

    def get_node_service_addr(self):
        return None


class RayJobArgs(JobArgs):
    def __init__(self, platform, namespace, job_name):
        super(RayJobArgs, self).__init__(platform, namespace, job_name)
        self.file_path = "{}.json".format(job_name)
        foler_path = os.path.dirname(os.path.dirname(__file__))
        self.file_path = os.path.join(foler_path, "tests/test.json")
        self.stats_backend = LocalFileStateBackend(self.file_path)

    def initilize(self):
        job = self.stats_backend.load()
        for replica, spec in job["spec"]["replicaSpecs"].items():
            num = int(spec.get("replicas", 0))
            requests = spec.get("resources", {})
            cpu = convert_cpu_to_decimal(requests.get("cpu", 0))
            if "memory" in requests:
                memory = convert_memory_to_mb(requests["memory"])
            else:
                memory = 0
            gpu_type = None
            gpu_num = 0
            for k, v in requests.items():
                if "nvidia.com" in k:
                    gpu_type = k
                    gpu_num = int(v)
            group_resource = NodeGroupResource(
                num,
                NodeResource(cpu, memory, gpu_type, gpu_num),
            )
            restart_count = int(spec.get("restartCount", 3))
            auto_scale = parse_bool(spec.get("autoScale", "True"))
            restart_timeout = int(spec.get("restartTimeout", 0))
            critical_nodes = spec.get("criticalNodes", "")
            self.node_args[replica] = NodeArgs(
                group_resource,
                auto_scale,
                restart_count,
                restart_timeout,
                critical_nodes,
            )
