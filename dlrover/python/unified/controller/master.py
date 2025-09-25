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

from typing import Dict, List

import ray
import ray.actor
from ray.exceptions import GetTimeoutError

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import ActorInfo
from dlrover.python.unified.common.enums import ExecutionResult, MasterStage

from ..common.config import JobConfig
from .api import (
    MASTER_ACTOR_NAME,
    MasterStatus,
    PrimeMasterApi,
)
from .manager import PrimeManager


class PrimeMaster:
    """The master actor for managing the job execution."""

    def __init__(self, config: JobConfig):
        assert (
            ray.get_runtime_context().get_actor_name() == MASTER_ACTOR_NAME
        ), (
            f"PrimeMaster must be initialized as a Ray actor "
            f"with the name '{MASTER_ACTOR_NAME}'."
        )

        self.manager = PrimeManager(config)
        if ray.get_runtime_context().was_current_actor_reconstructed:
            self.manager.self_recover()

    def get_status(self):
        """Get the current status of the job."""
        internal_state = self.manager.state
        return MasterStatus(
            internal_state.stage,
            internal_state.exit_code,
            internal_state.job_restart_count,
        )

    async def start(self):
        """Start the job execution."""
        await self.manager.prepare()
        await self.manager.start()

    async def stop(self):
        """Stop the job execution."""
        self.manager.request_stop("Requested stop.")

    async def wait(self):
        """Wait for the job to finish."""
        await self.manager.wait()

    async def shutdown(self):
        """Shutdown the master actor and clean up resources."""
        if self.manager.stage != MasterStage.STOPPED:
            logger.warning(
                f"Job is not stopped yet, current stage: {self.manager.stage}. "
            )
        ray.actor.exit_actor()

    # region RPC

    def get_actor_info(self, name: str) -> ActorInfo:
        """Get a actor by name."""
        actor = self.manager.graph.by_name.get(name)
        if actor is None:
            raise ValueError(f"Actor {name} not found.")
        return actor.to_actor_info()

    def get_workers_by_role(
        self, role: str, optional: bool = False
    ) -> List[ActorInfo]:
        """Get all actors by role."""
        role_info = self.manager.graph.roles.get(role)
        if role_info is None:
            if optional:
                return []
            raise ValueError(f"Role {role} not found.")
        return [node.to_actor_info() for node in role_info.instances]

    def get_all_roles(self) -> Dict[str, List[ActorInfo]]:
        """Get all roles."""
        return {
            role: [node.to_actor_info() for node in role_info.instances]
            for role, role_info in self.manager.graph.roles.items()
        }

    async def restart_actors(self, actor_names: List[str]) -> None:
        """Restart the specified actors by names."""
        assert all(
            actor in self.manager.graph.by_name for actor in actor_names
        ), f"Some actors {actor_names} not found in the graph."
        actors = [self.manager.graph.by_name[name] for name in actor_names]

        await self.manager.restart_actors(actors)

    async def restart(self):
        logger.info("Restarting the entire job by request.")
        await self.manager.restart_job()

    async def report_actor_restarted(self, name: str):
        actor = self.manager.graph.by_name.get(name)
        if actor is None:
            raise ValueError(f"Actor {name} not found.")

        await self.manager.deal_with_actor_restarting(actor)

    def register_data_queue(self, name: str, owner_actor: str, size: int):
        """Register a data queue."""
        self.manager.sync_manager.register_data_queue(name, owner_actor, size)

    async def get_data_queue_owner(self, name: str) -> str:
        """Get the owner actor of a data queue. Waits if not available."""
        return await self.manager.sync_manager.get_data_queue_owner(name)

    async def report_execution_result(
        self, name: str, result: ExecutionResult
    ):
        """Report the execution result of an actor."""
        actor = self.manager.graph.by_name.get(name)
        if actor is None:
            raise ValueError(f"Actor {name} not found.")
        await self.manager.deal_with_actor_finished(actor, result)

    # endregion
    @staticmethod
    def create(
        config: JobConfig, detached: bool = True
    ) -> "type[PrimeMasterApi]":
        """Create a PrimeMaster instance."""
        if not ray.is_initialized():
            logger.info("Ray is not initialized, initializing now.")
            ray.init()

        master_ud_resource = {}
        if config.master_isolation_schedule_resource:
            master_ud_resource[config.master_isolation_schedule_resource] = 1
            logger.info(
                f"Setup master actor isolation ud resource: {master_ud_resource}"
            )

        ref = (
            ray.remote(PrimeMaster)
            .options(
                name=MASTER_ACTOR_NAME,
                lifetime="detached" if detached else "normal",
                num_cpus=config.master_cpu,
                memory=config.master_mem,
                resources=master_ud_resource,
                max_restarts=config.master_max_restart,
                runtime_env={"env_vars": config.dl_config.global_envs},
            )
            .remote(config)
        )
        try:
            ray.get(
                ref.__ray_ready__.remote(),  # type: ignore
                timeout=config.master_create_timeout,
            )
        except GetTimeoutError:
            raise TimeoutError(
                f"Timeout waiting for PrimeMaster to be ready after {config.master_create_timeout} seconds."
            )
        return PrimeMasterApi
