# Copyright 2025 The DLRover Authors. All rights reserved.
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
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import (
    DLMasterConstant,
    DLWorkloadEnv,
)
from dlrover.python.unified.common.job_context import get_job_context
from dlrover.python.unified.master.graph import (
    DLExecutionGraph,
    DLExecutionVertex,
)
from dlrover.python.unified.master.placement import (
    SingleBundlePerNodePlacement,
    SingleGroupPerNodePlacement,
)

_job_ctx = get_job_context()


class Scheduler(ABC):
    """
    Scheduler is used to manage workload actor's lifecycle.
    """

    def __init__(self, execution_graph: DLExecutionGraph):
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
            try:
                return ray.get_runtime_context().current_actor
            except RuntimeError:
                logger.error("Failed to get master actor handle.")
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

    def _create_actor_by_graph(self):
        start = time.time() * 1000
        master_handle = self.get_master_actor_handle()
        config = self.graph.dl_config

        vertices = self.graph.get_all_vertices()

        for vertex in vertices:
            if vertex:
                actor_handle = self.__create_actor_by_vertex(
                    master_handle, vertex, config
                )
                logger.info(f"Creating workload actor: {vertex.name}")
                self.graph.update_actor_handle_for_vertex(actor_handle, vertex)

        return start

    def __get_runtime_env(self, vertex: DLExecutionVertex):
        env_key = "env_vars"

        # runtime env
        runtime_env = {
            env_key: {
                DLWorkloadEnv.JOB: _job_ctx.job_config.job_name,
                DLWorkloadEnv.NAME: vertex.name,
                DLWorkloadEnv.ROLE: vertex.role,
                DLWorkloadEnv.RANK: str(vertex.rank),
                DLWorkloadEnv.WORLD_SIZE: str(vertex.world_size),
                DLWorkloadEnv.LOCAL_RANK: str(vertex.local_rank),
                DLWorkloadEnv.LOCAL_WORLD_SIZE: str(vertex.local_world_size),
            }
        }

        # setup global env
        runtime_env[env_key].update(self.graph.dl_context.env)

        # setup role env
        runtime_env[env_key].update(
            self.graph.dl_context.workloads[vertex.role].instance_env
        )

        # setup device collocation env
        if vertex.get_core_resource_num() < 1:
            group_env_value = ""
            for group_tuple in self.graph.dl_context.workload_group.groups:
                group_roles = list(group_tuple[0].keys())
                if vertex.role in group_roles:
                    group_env_value = ",".join(r for r in group_roles)
                    break
            runtime_env[env_key][
                DLWorkloadEnv.DEVICE_COLLOCATION_GROUP
            ] = group_env_value

        # setup ray cuda visible env
        if not set(
            DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS.items()
        ).issubset(set(runtime_env[env_key].items())):
            # this env is used for disable 'ray set visible device' so we can
            # specify device by local_rank on ray(otherwise ray will assign a
            # specified device)
            runtime_env[env_key].update(
                DLWorkloadEnv.RAY_NOSET_VISIBLE_DEVICES_ENVS
            )
        else:
            # remove 'false' value setting for using 'ray set visible device'
            for key in DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS:
                runtime_env[env_key].pop(key, None)

        logger.debug(f"Create workload actor with runtime-env: {runtime_env}")

        return runtime_env

    def __create_actor_by_vertex(
        self, master_handle, vertex: DLExecutionVertex, config
    ) -> ActorHandle:

        if vertex.use_pg():
            actor_creation_opts = vertex.get_cls().options(
                name=vertex.name,
                lifetime="detached",
                num_cpus=vertex.resource.cpu,
                memory=vertex.resource.memory,
                num_gpus=vertex.resource.gpu,
                max_restarts=-1,
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
                max_restarts=-1,
                runtime_env=self.__get_runtime_env(vertex),
            )

        return actor_creation_opts.remote(
            master_handle,
            config,
        )

    def _check_actor_creation(self, start):
        # ping actor by reporting
        timeout = max(
            DLMasterConstant.SCHEDULING_TIMEOUT_MIN_SECS,
            len(self.graph.get_all_actor_handles())
            * DLMasterConstant.SCHEDULING_TIMEOUT_PER_ACTOR_SECS,
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

    def __init__(self, execution_graph: DLExecutionGraph):
        super().__init__(execution_graph)
        self._placement = self._get_placement()

    def _get_placement(self):
        if self.graph.dl_context.workload_group.has_device_collocate():
            logger.info(
                "Use 'SingleGroupPerNodePlacement' for workload group "
                "has device collocation."
            )
            return SingleGroupPerNodePlacement(self.graph)
        else:
            logger.info(
                "Use 'SingleBundlePerNodePlacement' for workload "
                "group has no device collocation."
            )
            return SingleBundlePerNodePlacement(self.graph)

    @property
    def placement(self):
        return self._placement

    def schedule(self):
        # prepare pg
        self.placement.prepare_placement_group()

        # create pg
        # note: need to create placement group first,
        # so we can pass the pg object to vertex for actor creation
        self.placement.create_placement_group()

        # allocate pg to actor
        self.placement.allocate_placement_group()

        # do creation
        start = self._create_actor_by_graph()

        # check creation
        self._check_actor_creation(start)
