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


class NodeResource(object):
    """NodeResource records a resource of a Node.
    Attributes:
        cpu: float, CPU cores.
        memory: float, memory MB.
        gpu: Dict.
    """

    def __init__(self, cpu, memory, gpu_type=None, gpu_num=0):
        self.cpu = cpu
        self.memory = memory
        self.gpu_type = gpu_type
        self.gpu_num = gpu_num

    @classmethod
    def resource_str_to_node_resource(cls, resource_str):
        """Convert the resource configuration like "memory=100Mi,cpu=5"
        to a NodeResource instance."""
        resource = {}
        if not resource_str:
            return resource
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


class NodeGroupResource(object):
    """The node group resource contains the number of the task
    and resource (cpu, memory) of each task.
    Args:
        count: int, the number of task.
        node_resource: a NodeResource instance.
    """

    def __init__(self, count, node_resource, priority=None):
        self.count = count
        self.node_resource = node_resource
        self.priority = priority
