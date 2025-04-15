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
import typing
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import ray
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from dlrover.python.common import env_utils
from dlrover.python.common.enums import ResourceType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.constant import RLMasterConstant, RLWorkloadEnv
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
        config = self.graph.rl_config

        vertices = self.graph.get_all_vertices()
        for vertex in vertices:
            if vertex:
                actor_handle = self.__create_actor_by_vertex(
                    master_handle, vertex, config
                )
                logger.info(f"Creating workload actor: {vertex.name}")
                self.graph.update_actor_handle_for_vertex(actor_handle, vertex)

        return start

    def __get_runtime_env(self, vertex: RLExecutionVertex):
        # runtime env
        runtime_env = {
            "env_vars": {
                RLWorkloadEnv.NAME: vertex.name,
                RLWorkloadEnv.ROLE: vertex.role.name,
                RLWorkloadEnv.RANK: str(vertex.rank),
                RLWorkloadEnv.WORLD_SIZE: str(vertex.world_size),
                RLWorkloadEnv.LOCAL_RANK: str(vertex.local_rank),
                RLWorkloadEnv.LOCAL_WORLD_SIZE: str(vertex.local_world_size),
                # this env is mandatory so we can specify device by local_rank
                # on ray(otherwise ray will assign a specified device)
                RLWorkloadEnv.RAY_NOSET_CUDA: "true",
            }
        }

        runtime_env["env_vars"].update(self.graph.rl_context.env)
        runtime_env["env_vars"].update(
            self.graph.rl_context.workloads[vertex.role].instance_env
        )
        logger.debug(f"Create workload actor with runtime-env: {runtime_env}")

        return runtime_env

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
                runtime_env=self.__get_runtime_env(vertex),
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
                runtime_env=self.__get_runtime_env(vertex),
            )

        return actor_creation_opts.remote(
            master_handle,
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
        ready, not_ready = ray.wait(
            ping_refs,
            num_returns=len(ping_refs),
            timeout=timeout,
        )
        if len(not_ready) > 0:
            raise TimeoutError(
                f"{len(not_ready)} workload actor "
                f"creation timeout: {timeout}s."
            )
        logger.info(
            f"{len(ready)} workload actors created, "
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

    SINGLE_PG_NAME = "SINGLE_PG_WITH_SINGLE_BUNDLE_PER_NODE"

    def schedule(self):
        # prepare pg
        self._prepare_placement_group()

        # create pg
        # note: need to create placement group first,
        # so we can pass the pg object to vertex for actor creation
        self._create_placement_group()

        # allocate pg to actor
        self._allocate_placement_group()

        # do creation
        start = self._create_actor_by_graph(with_pg=True)

        # check creation
        self._check_actor_creation(start)

    # TODO: abstraction on pg operation: prepare, allocate, create

    def _get_placement_strategy(self):
        strategy_from_env = env_utils.get_env(RLMasterConstant.PG_STRATEGY_ENV)

        if not strategy_from_env:
            return "STRICT_SPREAD"
        return strategy_from_env

    # prepare placement group
    def _prepare_placement_group(self):
        # single pg and single bundle per node

        strategy = self._get_placement_strategy()

        # define bundle unit resource
        device_per_node = self.graph.rl_context.trainer.device_per_node
        if self.graph.rl_context.trainer.device_type == ResourceType.CPU:
            bundle_unit_resource = {"CPU": device_per_node}
        else:
            bundle_unit_resource = {"GPU": device_per_node}

        # create bundle by nodes number
        bundles = []
        for i in range(self.graph.rl_context.trainer.node_number):
            bundles.append(bundle_unit_resource)

        logger.info(f"Prepare placement group with bundles: {bundles}")
        self.graph.add_placement_group(
            GroupOrderedScheduler.SINGLE_PG_NAME,
            PlacementGroupAllocation(
                GroupOrderedScheduler.SINGLE_PG_NAME, 0, strategy, bundles
            ),
        )

    def __get_start_bundle(self, allocated_bundles, bundle_topology) -> int:
        if not allocated_bundles:
            return 0

        counter: typing.Counter[int] = Counter(allocated_bundles)
        max_bundle_index = max(counter.keys())
        for bundle_allocated in list(counter.items()):
            if bundle_allocated[1] >= bundle_topology[bundle_allocated[0]]:
                continue
            return int(bundle_allocated[0])

        return max_bundle_index + 1

    # allocate placement group
    def _allocate_placement_group(self):
        workload_group = self.graph.get_workload_group()
        pg = self.graph.get_placement_group()[
            GroupOrderedScheduler.SINGLE_PG_NAME
        ][0]
        allocated_bundles = []
        bundle_topology = self.graph.get_bundle_topology()

        for group_desc_tuple in workload_group.groups:
            group_dict = group_desc_tuple[0]

            for role, role_group_size in group_dict.items():
                bundle_index = self.__get_start_bundle(
                    allocated_bundles, bundle_topology
                )
                i = 0
                for vertex in self.graph.execution_vertices[role]:
                    vertex.update_pg_info(pg.pg_instance, bundle_index)
                    allocated_bundles.append(bundle_index)
                    pg.allocate(vertex.name, bundle_index)

                    i += 1
                    if i == role_group_size:
                        bundle_index += 1

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
