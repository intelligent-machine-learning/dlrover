# Copyright 2022 The DLRover Authors. All rights reserved.
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

import copy
import time

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeResourceLimit,
    NodeStatus,
    PriorityClass,
)
from dlrover.python.common.grpc import ParallelConfig
from dlrover.python.common.serialize import JsonSerializable


def _is_float_str(str_number):
    if not str_number:
        return False
    try:
        float(str_number)
        return True
    except ValueError:
        return False


class NodeResource(JsonSerializable):
    """NodeResource records a resource of a Node.
    Attributes:
        cpu: float, CPU cores.
        memory: float, memory MB.
        gpu_type: str, the type of GPU.
        gpu_num: int,
        gpu_stats: list of GPUMetric, a list of GPUMetric dataclass objects,
        each containing GPU statistics.
            - index (int): The index of the GPU.
            - total_memory_mb (int): Total GPU memory in megabytes.
            - used_memory_mb (int): Used GPU memory in megabytes.
            - gpu_utilization (float): GPU utilization in percentage (0.0 to
              100.0).
        image: the image name of the node.
        priority: the priority classs of the node.
    Example:
    To create an instance of NodeResource with the following attributes:
    - cpu: 4.0
    - memory: 8192
    - gpu_type: "nvidia.com"
    - gpu_num: 1
    - gpu_stats: [GPUMetric(index=0, total_memory_mb=8192, used_memory_mb=2048,
    gpu_utilization=80.0)]
    - image: "ubuntu:20.04"
    - priority: "high"

    >>> resource = NodeResource(
    ...     cpu=4.0,
    ...     memory=8192,
    ...     gpu_type="nvidia.com",
    ...     gpu_num=1,
    ...     gpu_stats=[GPUStats(index=0, total_memory_mb=8192,
    ...     used_memory_mb=2048, gpu_utilization=80.0)],
    ...     image="ubuntu:20.04",
    ...     priority="high"
    ... )
    """

    def __init__(
        self,
        cpu,
        memory,
        gpu_type="",
        gpu_num=0,
        gpu_stats=[],
        priority="",
        **kwargs,
    ):
        self.cpu = cpu
        self.memory = memory
        self.gpu_type = gpu_type
        self.gpu_num = gpu_num
        self.gpu_stats = gpu_stats
        self.kwargs = kwargs
        self.image = ""
        self.priority = priority

    def to_resource_dict(self):
        resource = self.kwargs
        resource["cpu"] = self.cpu
        resource["memory"] = str(self.memory) + "Mi"
        if self.gpu_num > 0:
            resource[self.gpu_type] = self.gpu_num
        return resource

    @classmethod
    def resource_str_to_node_resource(cls, resource_str):
        """Convert the resource configuration like "memory=100Mi,cpu=5"
        to a NodeResource instance."""
        resource = {}
        if not resource_str:
            return NodeResource(0, 0)
        for value in resource_str.strip().split(","):
            resource[value.split("=")[0]] = value.split("=")[1]

        memory = float(resource.get("memory", "0Mi")[0:-2])
        cpu = float(resource.get("cpu", "0"))
        gpu_type = None
        gpu_num = 0
        for key, _ in resource.items():
            if "nvidia.com" in key:
                gpu_type = key
                gpu_num = int(resource[key])
        return NodeResource(cpu, memory, gpu_type, gpu_num)


class NodeGroupResource(JsonSerializable):
    """The node group resource contains the number of the task
    and resource (cpu, memory) of each task.
    Args:
        count: int, the number of task.
        node_resource: a NodeResource instance.
    """

    def __init__(self, count, node_resource: NodeResource):
        self.count = count
        self.node_resource = node_resource

    def update(self, count, cpu, memory):
        if count > 0:
            self.count = count
        if cpu > 0:
            self.node_resource.cpu = cpu
        if memory > 0:
            self.node_resource.memory = memory

    @classmethod
    def new_empty(cls):
        return NodeGroupResource(0, NodeResource(0, 0))


class Node(object):
    """Node records the information of each training node.
    Attributes:
        type: str, the type (e.g. "ps", "worker") of a node.
        id: int, the id of a node.
        name: str, the name of a node.
        status: the status of a node.
        start_time: int, the start timestamp of a node.
        rank_index: int, the rank index of a node in a training cluster.
        relaunch_count: int, the relaunched number of the training node.
        critical: bool, if true, the job will fail if the node fails.
        max_relaunch_count: int, the maximum to relaunch a node.
        relaunchable: bool, whether to relaunch a node if it fails.
        is_released: bool, ture if the master deletes the node.
        exit_reason: str, the exited reason of a node.
        used_resource: the resource usage of the node.
        init_time: the timestamp to initialize the node object.
        host_name: the name of the host where the node is placed.
        host_ip: the ip of host node.
        unrecoverable_failure_msg: unrecoverable failure msg.
    """

    def __init__(
        self,
        node_type,
        node_id,
        config_resource: NodeResource = NodeResource(0, 0),
        name=None,
        status=NodeStatus.INITIAL,
        start_time=None,
        rank_index=None,
        relaunch_count=0,
        critical=False,
        max_relaunch_count=0,
        relaunchable=True,
        service_addr=None,
        host_name=None,
        host_ip=None,
        paral_config=ParallelConfig(),
        restart_training=False,
    ):
        self.type = node_type
        self.id = node_id
        self.name = name
        self.status = status
        self.start_time = start_time
        self.rank_index = rank_index if rank_index is not None else node_id
        self.relaunch_count = relaunch_count
        self.critical = critical
        self.max_relaunch_count = max_relaunch_count
        self.relaunchable = relaunchable
        self.service_addr = service_addr
        self.create_time = None
        self.finish_time = None
        self.is_recovered_oom = False
        self.is_released = False
        self.exit_reason = ""
        self.config_resource = config_resource
        self.used_resource = NodeResource(0.0, 0.0)
        self.start_hang_time = 0
        self.init_time = time.time()
        self.eval_time = 0
        self.host_name = host_name
        self.host_ip = host_ip
        self.hang = False
        self.paral_config = paral_config
        self.restart_training = restart_training
        self.migrated = False
        self.unrecoverable_failure_msg = ""
        self.heartbeat_time = 0

    def exited(self):
        return self.status in [
            NodeStatus.FAILED,
            NodeStatus.SUCCEEDED,
            NodeStatus.FINISHED,
        ]

    def update_info(
        self,
        name=None,
        start_time=None,
        create_time=None,
        host_name=None,
        host_ip=None,
        restart_training=False,
        relaunch_count=0,
    ):
        if name is not None:
            self.name = name
        if start_time is not None:
            self.start_time = start_time
        if create_time is not None:
            self.create_time = create_time
        if host_name:
            self.host_name = host_name
        if host_ip:
            self.host_ip = host_ip
        self.relaunch_count = max(self.relaunch_count, relaunch_count)
        self.restart_training = restart_training

    def update_status(self, status=None):
        if status is not None:
            self.status = status

    def update_resource_usage(self, cpu, memory, gpu_stats=[]):
        self.used_resource.cpu = round(cpu, 2)
        self.used_resource.memory = memory
        self.used_resource.gpu_stats = gpu_stats

    def update_paral_config(self, paral_config):
        self.paral_config = paral_config

    def update_service_address(self, service_addr):
        self.service_addr = service_addr

    def get_relaunch_node_info(self, new_id):
        new_node = copy.deepcopy(self)
        new_node.id = new_id
        new_node.name = None
        new_node.status = NodeStatus.INITIAL
        new_node.start_time = None
        new_node.create_time = None
        new_node.finish_time = None
        new_node.is_released = False
        new_node.relaunchable = True
        new_node.init_time = time.time()
        return new_node

    def is_unrecoverable_failure(self):
        cpu_memory_overload = (
            self.config_resource.gpu_num == 0
            and self.config_resource.memory >= NodeResourceLimit.MAX_MEMORY
            and self.exit_reason == NodeExitReason.OOM
        )
        if self.relaunch_count >= self.max_relaunch_count:
            self.unrecoverable_failure_msg = (
                "exhausted {} relaunch opportunities".format(
                    self.max_relaunch_count
                )
            )
            return True

        if self.exit_reason == NodeExitReason.FATAL_ERROR:
            self.unrecoverable_failure_msg = "fatal error"
            return True

        if cpu_memory_overload:
            self.unrecoverable_failure_msg = (
                "oom error and can not add more memory"
            )
            return True

        return False

    def set_exit_reason(self, reason):
        self.exit_reason = reason

    def update_priority(self, group_node_num):
        """Update the priority if the priority is a fraction.
        For example, if the priority is 0.5, and the number of
        typed nodes is 10. The node priority with id <5 is high
        and others are low.
        Args:
            group_node_num: the number of the group nodes.
        """
        priority = self.config_resource.priority
        if _is_float_str(priority):
            fraction = float(priority)
            if fraction <= 0 or fraction > 1:
                raise ValueError(
                    "If priority is a float, it should be greater than 0 or"
                    "less equal than 1."
                )
            high_count = round(group_node_num * fraction)
            if self.id <= high_count:
                self.config_resource.priority = PriorityClass.HIGH
            else:
                self.config_resource.priority = PriorityClass.LOW
        elif priority not in [None, "", PriorityClass.HIGH, PriorityClass.LOW]:
            raise ValueError(
                "Not support priority = {}, please set priority = "
                "high/low/a fraction value.".format(priority)
            )

    def timeout(self, timeout):
        now = time.time()
        if (
            now - self.init_time > timeout
            and self.status == NodeStatus.INITIAL
        ):
            return True

    def __repr__(self):
        return (
            f"name:{self.name};"
            f"rank_index:{self.rank_index};"
            f"type:{self.type};"
            f"status:{self.status};"
            f"addr:{self.service_addr};"
            f"is_released:{self.is_released};"
            f"priroity:{self.config_resource.priority}"
        )

    def to_dict(self):
        d = copy.deepcopy(self.__dict__)
        d.pop("paral_config", None)
        d.pop("config_resource", None)
        d.pop("used_resource", None)
        return d
