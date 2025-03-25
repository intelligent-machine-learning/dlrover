# Copyright 2025 The EasyDL Authors. All rights reserved.
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
from dlrover.python.rl.master.execution.graph import RLExecutionGraph


class Executor(object):
    def __init__(self, execution_graph: RLExecutionGraph):
        self.__execution_graph = execution_graph

    def create_workloads(self):
        pass

    def destroy_workloads(self):
        pass
