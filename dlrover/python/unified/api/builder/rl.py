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
from pydantic import model_validator

from dlrover.python.unified.common.enums import (
    DLStreamType,
    RLRoleType,
)

from .base import DLJob, DLJobBuilder


class RLJob(DLJob):
    @model_validator(mode="after")
    def validate(self):
        if self.stream_type != DLStreamType.TASK_STREAM:
            raise ValueError(
                "Invalid stream type for RLJob, expected TASK_STREAM"
            )

        if RLJobBuilder.ACTOR_ROLE not in self.workloads:
            raise ValueError("'actor' must be configured.")

        for role, _ in self.workloads.items():
            if role not in RLJobBuilder.ROLES:
                raise ValueError(
                    f"Invalid role '{role}' for RLJob, "
                    f"supported roles are {RLJobBuilder.ROLES}."
                )
        return self


class RLJobBuilder(DLJobBuilder):
    """
    Extension job builder for Reinforcement Learning (RL).
    """

    TRAINER_ROLE = RLRoleType.TRAINER.name
    ACTOR_ROLE = RLRoleType.ACTOR.name
    REF_ROLE = RLRoleType.REFERENCE.name
    REW_ROLE = RLRoleType.REWARD.name
    CRITIC_ROLE = RLRoleType.CRITIC.name
    ROLLOUT_ROLE = RLRoleType.ROLLOUT.name
    ROLES = [
        TRAINER_ROLE,
        ACTOR_ROLE,
        REF_ROLE,
        REW_ROLE,
        CRITIC_ROLE,
        ROLLOUT_ROLE,
    ]

    def __init__(self):
        super(RLJobBuilder, self).__init__()

    def trainer(self, entrypoint: str):
        """
        Setup trainer for user-defined task stream.

        Args:
            entrypoint (str): The entry point of actor.
        """

        builder = self.role(RLJobBuilder.TRAINER_ROLE).run(entrypoint)

        # default property
        builder.total(1)
        builder.per_group(1)
        builder.resource(cpu=4, mem=8192)

        return builder

    def actor(self, entrypoint: str):
        """
        Setup actor.

        Args:
            entrypoint (str): The entry point of actor.
        """

        return (
            self.role(RLJobBuilder.ACTOR_ROLE).train(entrypoint).not_driver()
        )

    def rollout(self, entrypoint: str):
        """
        Setup rollout.

        Args:
            entrypoint (str): The entry point of rollout.
        """
        return (
            self.role(RLJobBuilder.ROLLOUT_ROLE).run(entrypoint).not_driver()
        )

    def reference(self, entrypoint: str):
        """
        Setup reference.

        Args:
            entrypoint (str): The entry point of reference.
        """

        return self.role(RLJobBuilder.REF_ROLE).run(entrypoint).not_driver()

    def reward(self, entrypoint: str):
        """
        Setup reward.

        Args:
            entrypoint (str): The entry point of reward.
        """

        return self.role(RLJobBuilder.REW_ROLE).run(entrypoint).not_driver()

    def critic(self, entrypoint: str):
        """
        Setup critic.

        Args:
            entrypoint (str): The entry point of critic.
        """

        return (
            self.role(RLJobBuilder.CRITIC_ROLE).train(entrypoint).not_driver()
        )

    def with_collocation_all(self):
        """
        Set a collocation strategy for all roles.

        Notice: can be used after role definition only
        """

        super().with_collocation_all(RLRoleType.TRAINER.name)
        return self

    def build(self) -> RLJob:
        return RLJob.model_validate(super().build().model_dump())
