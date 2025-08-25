#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from abc import ABC
from typing import Dict

import ray

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.util.common_util import find_free_port_from_env


class BaseRLWorker(ActorBase, ABC):
    @property
    def role(self) -> str:
        return self.actor_info.role

    @property
    def rank(self) -> int:
        return self.actor_info.rank

    @property
    def world_size(self) -> int:
        return self.actor_info.spec.total

    @property
    def local_rank(self) -> int:
        return self.actor_info.local_rank

    @property
    def local_world_size(self) -> int:
        return self.actor_info.spec.per_group

    @property
    def config(self):
        return self.job_info.user_config

    @property
    def torch_master_addr(self) -> str:
        if DLWorkloadEnv.MASTER_ADDR in os.environ:
            return os.environ[DLWorkloadEnv.MASTER_ADDR]
        return ""

    @property
    def torch_master_port(self) -> int:
        if DLWorkloadEnv.MASTER_PORT in os.environ:
            return int(os.environ[DLWorkloadEnv.MASTER_PORT])
        return -1

    def get_device_collocation(self) -> str:
        return env_utils.get_env(DLWorkloadEnv.DEVICE_COLLOCATION_GROUP) or ""

    def has_device_collocation(self):
        return (
            self.get_device_collocation()
            and self.role in self.get_device_collocation()
        )

    def is_actor_role(self):
        return self.role == RLRoleType.ACTOR.name

    def is_rollout_role(self):
        return self.role == RLRoleType.ROLLOUT.name

    def is_reward_role(self):
        return self.role == RLRoleType.REWARD.name

    def is_ref_role(self):
        return self.role == RLRoleType.REFERENCE.name

    def is_critic_role(self):
        return self.role == RLRoleType.CRITIC.name

    def is_actor_or_rollout_device_collocation(self):
        return (
            (self.is_actor_role() or self.is_rollout_role())
            and RLRoleType.ACTOR.name in self.get_device_collocation()
            and RLRoleType.ROLLOUT.name in self.get_device_collocation()
        )

    # ray.remote

    def get_master_addr(self):
        """Get the master address for process group(for rank0 only)."""
        return ray.util.get_node_ip_address(), find_free_port_from_env()

    def setup_rl_workload(self, env_dict: Dict[str, str]) -> bool:
        """
        Internal function. Do not override.
        """

        # update envs
        for key, value in env_dict.items():
            os.environ[key] = value
            logger.info(f"Setup env: {key}-{value}")

        self._update_stage_force(WorkerStage.RUNNING, WorkerStage.READY)
        return True

    def update_rl_workload_stage(self, worker_stage: WorkerStage):
        self._update_stage_force(worker_stage)

    def start(self):
        """Only Trainer needs to implement this method."""
        pass
