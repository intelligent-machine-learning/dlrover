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
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.resource import Resource
from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.common.enums import RLRoleType, WorkloadGroupType
from dlrover.python.rl.common.exception import ResourceError
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.graph import (
    PlacementGroupAllocation,
    RLExecutionGraph,
    RLExecutionVertex,
)

_job_ctx = get_job_context()


class Scheduler(ABC):
    """
    Scheduler is used to manage workload actor's lifecycle.
    """

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
        # remove actors
        start = time.time() * 1000
        futures = []
        vertices = self.graph.get_all_vertices()
        max_workers = max(int(len(vertices) / 4), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for vertex in vertices:
                futures.append(executor.submit(vertex.cleanup))

            # wait all result
            for future in futures:
                future.result()

        logger.info(
            f"{len(vertices)} workload actors are removed, "
            f"cost: {time.time() * 1000 - start:.2f}ms."
        )

        # remove placement groups
        if self.graph.get_placement_group():
            self.graph.remove_placement_group()
            self.graph.cleanup_placement_group_allocation()
            logger.info("Placement group is removed and allocation is reset.")

    def _create_actor_by_graph(self, with_pg=False):
        start = time.time() * 1000
        master_handle = self.get_master_actor_handle()
        config = self.graph.get_rl_config()

        if with_pg:
            # create actor with pg 1st then others
            vertices = self.graph.get_all_vertices_with_pg_priority()
        else:
            # create actor directly
            vertices = self.graph.get_all_vertices()

        for vertex in vertices:
            if vertex:
                actor_handle = self.__create_actor_by_vertex(
                    master_handle, vertex, config
                )
                logger.info(f"Creating workload actor: {vertex.name}")
                self.graph.update_actor_handle_for_vertex(actor_handle, vertex)

        return start

    def __create_actor_by_vertex(
        self, master_handle, vertex: RLExecutionVertex, config
    ) -> ActorHandle:

        if vertex.use_pg():
            actor_creation_opts = vertex.get_cls().options(
                name=vertex.name,
                lifetime="detached",
                num_cpus=vertex.resource.cpu,
                memory=vertex.resource.memory,
                num_gpus=vertex.resource.gpu,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=vertex.pg,
                    placement_group_bundle_index=vertex.pg_bundle_index,
                ),
            )
        else:
            actor_creation_opts = vertex.get_cls().options(
                name=vertex.name,
                lifetime="detached",
                num_cpus=vertex.resource.cpu,
                memory=vertex.resource.memory,
                num_gpus=vertex.resource.gpu,
            )

        return actor_creation_opts.remote(
            master_handle,
            vertex.name,
            vertex.role,
            vertex.rank,
            vertex.world_size,
            config,
        )

    def _check_actor_creation(self, start):
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
            f"{len(readies)} workload actors created, "
            f"cost: {time.time() * 1000 - start:.2f}ms."
        )


class SimpleScheduler(Scheduler):
    """
    Schedule workload actor async directly according to the vertices in
    execution graph.
    """

    def schedule(self):
        # do creation
        start = self._create_actor_by_graph()

        # check creation
        self._check_actor_creation(start)


class GroupOrderedScheduler(Scheduler):
    """
    Schedule workload actor group by group according to the order express
    by resource-group in execution graph.
    """

    def schedule(self):
        # prepare pg
        self._prepare_placement_group()

        # create pg
        # note: need to create placement group first,
        # so we can pass the pg object to vertex for actor creation
        self._create_placement_group()

        # allocate pg to actor
        self.graph.allocate_placement_group()

        # do creation
        start = self._create_actor_by_graph(with_pg=True)

        # check creation
        self._check_actor_creation(start)

    # prepare placement group
    def _prepare_placement_group(self):
        for group_type, groups in self.graph.get_workload_groups().items():
            if (
                group_type.name == WorkloadGroupType.HOST_GROUP.name
                or group_type.name == WorkloadGroupType.DEVICE_GROUP.name
            ):
                strategy = "STRICT_PACK"
            elif group_type.name == WorkloadGroupType.ANTI_HOST_GROUP.name:
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
                        bundles.setdefault(role, []).append(
                            (bundle_index, resource)
                        )
                        bundle_index += 1

                # create pg allocation
                pg_name = group_desc.get_group_name()
                logger.info(
                    f"Prepare {pg_num} placement group, "
                    f"group name: {pg_name}, strategy: {strategy}, "
                    f"bundles(size:{bundle_index}): {bundles}"
                )
                for pg_index in range(pg_num):
                    self.graph.add_placement_group(
                        pg_name,
                        PlacementGroupAllocation(
                            pg_name, pg_index, strategy, bundles
                        ),
                    )

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
