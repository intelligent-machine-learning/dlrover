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

from typing import List

import ray
import ray.actor

from dlrover.python.unified.common.workload_base import ActorInfo
from dlrover.python.unified.util.test_hooks import init_coverage

from .api import (
    MASTER_ACTOR_ID,
    MasterStatus,
    PrimeMasterApi,
    PrimeMasterRemote,
)
from .config import JobConfig
from .manager import PrimeManager

init_coverage()  # support coverage for master actor


class PrimeMaster(PrimeMasterRemote):
    def __init__(self, config: JobConfig):
        assert ray.get_runtime_context().get_actor_name() == MASTER_ACTOR_ID, (
            f"PrimeMaster must be initialized as a Ray actor "
            f"with the name '{MASTER_ACTOR_ID}'."
        )

        self.manager = PrimeManager(config)

    def get_status(self):
        return MasterStatus(stage=self.manager.stage)

    async def start(self):
        await self.manager.prepare()
        await self.manager.start()

    async def stop(self):
        await self.manager.stop()

    async def shutdown(self):
        ray.actor.exit_actor()

    # region RPC

    def get_actor_info(self, name: str) -> ActorInfo:
        """Get a actor by name."""
        actor = self.manager.graph.by_name.get(name)
        if actor is None:
            raise ValueError(f"Actor {name} not found.")
        return actor.to_actor_info()

    def get_workers_by_role(self, role: str) -> List[ActorInfo]:
        """Get all actors by role."""
        role_info = self.manager.graph.roles.get(role)
        if role_info is None:
            raise ValueError(f"Role {role} not found.")
        return [node.to_actor_info() for node in role_info.instances]

    @staticmethod
    def create(
        config: JobConfig, detached: bool = True
    ) -> "PrimeMasterRemote":
        """Create a PrimeMaster instance."""
        ref = (
            ray.remote(PrimeMaster)
            .options(
                name=MASTER_ACTOR_ID,
                lifetime="detached" if detached else "normal",
                num_cpus=config.master_cpu,
                memory=config.master_mem,
                max_restarts=config.master_max_restart,
                runtime_env={"env_vars": config.dl_config.global_envs},
            )
            .remote(config)
        )
        ray.get(ref.__ray_ready__.remote())
        return PrimeMasterApi

    # endregion
