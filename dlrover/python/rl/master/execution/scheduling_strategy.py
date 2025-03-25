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
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.master.execution.graph import RLExecutionGraph


class SchedulingStrategy(ABC):
    @abstractmethod
    def schedule(self, execution_graph: RLExecutionGraph):
        """Schedule workload actor by different strategy."""

    def cleanup(self, execution_graph: RLExecutionGraph):
        all_actor_handles = execution_graph.get_all_actor_handles()
        futures = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            for actor_handle in range(len(all_actor_handles)):
                futures.append(executor.submit(ray.kill, actor_handle))

            # wait all result
            for future in futures:
                future.result()


class SimpleOrderedStrategy(SchedulingStrategy):
    """
    Schedule workload actor one by one according to the order express
    by vertices directly in execution graph.
    """

    def schedule(self, execution_graph: RLExecutionGraph):
        pass


class GroupOrderedStrategy(SchedulingStrategy):
    """
    Schedule workload actor group by group according to the order express
    by resource-group in execution graph.
    """

    def schedule(self, execution_graph: RLExecutionGraph):
        pass
