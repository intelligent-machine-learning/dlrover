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
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import ray
from ray.actor import ActorHandle

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.execution.graph import (
    RLExecutionGraph,
    RLExecutionVertex,
)

_job_ctx = get_job_context()


class SchedulingStrategy(ABC):
    @abstractmethod
    def schedule(self, execution_graph: RLExecutionGraph):
        """Schedule workload actor by different strategy."""

    @classmethod
    def get_master_actor_handle(cls):
        if ray.is_initialized():
            return ray.get_runtime_context().current_actor
        return None

    def cleanup(self, execution_graph: RLExecutionGraph):
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for actor_handle in execution_graph.get_all_actor_handles():
                futures.append(executor.submit(ray.kill, actor_handle))

            # wait all result
            for future in futures:
                future.result()

    def create_actor_by_vertex(
        self, master_handle, vertex: RLExecutionVertex, config
    ) -> ActorHandle:
        return (
            vertex.get_cls()
            .options(
                name=vertex.name,
                lifetime="detached",
                num_cpus=vertex.resource.cpu,
                memory=vertex.resource.memory,
                num_gpus=vertex.resource.gpu,
            )
            .remote(
                master_handle,
                vertex.name,
                vertex.role,
                vertex.rank,
                vertex.world_size,
                config,
            )
        )


class SimpleStrategy(SchedulingStrategy):
    """
    Schedule workload actor async directly according to the vertices in
    execution graph.
    """

    def schedule(self, execution_graph: RLExecutionGraph):
        master_handle = self.get_master_actor_handle()
        config = execution_graph.get_rl_config()
        start = time.time() * 1000

        # create actor directly
        for vertex in execution_graph.get_all_vertices():
            if vertex:
                actor_handle = self.create_actor_by_vertex(
                    master_handle, vertex, config
                )
                logger.info(f"Creating workload actor: {vertex.name}")
                execution_graph.update_actor_handle_for_vertex(
                    actor_handle, vertex
                )

        # ping actor by reporting
        timeout = max(
            RLMasterConstant.SCHEDULING_TIMEOUT_MIN_SECS,
            len(execution_graph.get_all_actor_handles())
            * RLMasterConstant.SCHEDULING_TIMEOUT_PER_ACTOR_SECS,
        )

        ping_refs = [
            vertex.actor_handle.ping.remote()
            for vertex in execution_graph.get_all_vertices()
        ]
        readies, not_readies = ray.wait(
            ping_refs,
            num_returns=len(ping_refs),
            timeout=timeout,
        )
        if len(not_readies) > 0:
            raise TimeoutError(
                f"{len(not_readies)} workload actor "
                f"creation timeout: {timeout}s."
            )
        logger.info(
            f"{len(readies)} workloads created, "
            f"cost: {time.time() * 1000 - start:.2f}ms."
        )


class GroupOrderedStrategy(SchedulingStrategy):
    """
    Schedule workload actor group by group according to the order express
    by resource-group in execution graph.
    """

    def schedule(self, execution_graph: RLExecutionGraph):
        # TODO
        pass
