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
from typing import Dict, List, Tuple

import ray
from ray.exceptions import GetTimeoutError

from common.resource import Resource
from ray.actor import ActorHandle
from ray.util.placement_group import PlacementGroup
from rl.common.enums import RLRoleType, WorkloadGroupType
from rl.common.exception import ResourceError
from rl.common.rl_context import WorkloadGroupDesc

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.execution.graph import (
    RLExecutionGraph,
    RLExecutionVertex, PlacementGroupAllocation,
)

_job_ctx = get_job_context()


class SchedulingStrategy(ABC):

    def __init__(self, execution_graph: RLExecutionGraph):
        self._graph = execution_graph

    @property
    def graph(self):
        return self._graph

    @abstractmethod
    def schedule(self):
        """Schedule workload actor by different strategy."""

    @classmethod
    def get_master_actor_handle(cls):
        if ray.is_initialized():
            return ray.get_runtime_context().current_actor
        return None

    def cleanup(self):
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for actor_handle in self.graph.get_all_actor_handles():
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

    def schedule(self):
        master_handle = self.get_master_actor_handle()
        config = self.graph.get_rl_config()
        start = time.time() * 1000

        # create actor directly
        for vertex in self.graph.get_all_vertices():
            if vertex:
                actor_handle = self.create_actor_by_vertex(
                    master_handle, vertex, config
                )
                logger.info(f"Creating workload actor: {vertex.name}")
                self.graph.update_actor_handle_for_vertex(
                    actor_handle, vertex
                )

        # ping actor by reporting
        timeout = max(
            RLMasterConstant.SCHEDULING_TIMEOUT_MIN_SECS,
            len(self.graph.get_all_actor_handles())
            * RLMasterConstant.SCHEDULING_TIMEOUT_PER_ACTOR_SECS,
        )

        ping_refs = [
            vertex.actor_handle.ping.remote()
            for vertex in self.graph.get_all_vertices()
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

    def schedule(self):
        workload_groups = self.graph.get_workload_groups()

        # prepare pg
        for group_type, groups in workload_groups.items():
            self._prepare_placement_group(group_type, groups)

        # allocate pg
        self.graph.allocate_placement_group()

        # create pg
        self._create_placement_group()

        # create actors with pg


        # create actors without pg


    # prepare placement group
    def _prepare_placement_group(self, group_type: WorkloadGroupType, groups: List[WorkloadGroupDesc]):
        if (group_type == WorkloadGroupType.HOST_GROUP or
                group_type == WorkloadGroupType.DEVICE_GROUP):
            strategy = "STRICT_PACK"
        elif group_type == WorkloadGroupType.ANTI_HOST_GROUP:
            strategy = "STRICT_SPREAD"
        else:
            raise NotImplementedError()

        for group_desc in groups:
            # create pg one by one
            group_allocation: Dict[RLRoleType, int] = group_desc.allocation

            # 1st role and number in group
            role, number = next(iter(group_allocation.items()))

            # pg number to create
            pg_num = self.graph.get_workloads_size_by_role(role) / number
            # the validation is already done when building rl context
            assert pg_num == int(pg_num)
            pg_num = int(pg_num)

            # calculate bundles
            bundles: Dict[RLRoleType, List[Tuple[int, Resource]]] = {}
            bundle_index = 0
            for role, number in group_allocation.items():
                resource = self.graph.get_unit_resource_by_role(role)
                for index in range(number):
                    bundles.setdefault(role, []).append((bundle_index, resource))
                    bundle_index += 1

            # create pg allocation
            pg_name = group_desc.get_group_name()
            logger.info(f"Create {pg_num} placement group, "
                        f"group name: {pg_name}, strategy: {strategy}, "
                        f"bundles: {bundles}")
            for pg_index in range(pg_num):
                self.graph.add_placement_group(PlacementGroupAllocation(pg_name, pg_index, strategy, bundles))

    # create placement group by ray api
    def _create_placement_group(self):
        start = time.time()

        try:
            self.graph.create_placement_group()
        except GetTimeoutError:
            logger.error("Got timeout when creating placement group.")
            raise ResourceError()

        logger.info(
            f"All placement group created used: {time.time() - start:.2f}s"
        )
