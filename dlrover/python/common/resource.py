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
    def __init__(self, cpu, memory, gpu=None):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu


class TaskGroupResource(object):
    """The task group resource contains the number of the task
    and resource (cpu, memory) of each task.
    Args:
        count: int, the number of task.
        node_resource: a NodeResource instance.
    """

    def __init__(self, count, node_resource):
        self.count = count
        self.node_resource = node_resource
