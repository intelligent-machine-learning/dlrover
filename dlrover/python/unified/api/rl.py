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
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.api.base import DLJobBuilder
from dlrover.python.unified.common.enums import (
    DLStreamType,
    DLType,
    RLRoleType,
)


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
        self._dl_type = DLType.RL.name

    def trainer(self, module_name, class_name):
        """
        Setup trainer for user-defined task stream.

        Args:
            module_name (str): The module name of trainer.
            class_name (str): The class name of trainer.
        """

        assert self._stream_type == DLStreamType.TASK_STREAM

        builder = self.workload(
            RLJobBuilder.TRAINER_ROLE, module_name, class_name
        )

        # default property
        builder.total(1)
        builder.per_node(1)
        builder.resource(cpu=4, mem=8192)

        return builder

    def actor(self, module_name, class_name):
        """
        Setup actor.

        Args:
            module_name (str): The module name of actor.
            class_name (str): The class name of actor.
        """

        return self.workload(RLJobBuilder.ACTOR_ROLE, module_name, class_name)

    def rollout(self, module_name, class_name):
        """
        Setup rollout.

        Args:
            module_name (str): The module name of rollout.
            class_name (str): The class name of rollout.
        """

        return self.workload(
            RLJobBuilder.ROLLOUT_ROLE, module_name, class_name
        )

    def reference(self, module_name, class_name):
        """
        Setup reference.

        Args:
            module_name (str): The module name of reference.
            class_name (str): The class name of reference.
        """

        return self.workload(RLJobBuilder.REF_ROLE, module_name, class_name)

    def reward(self, module_name, class_name):
        """
        Setup reward.

        Args:
            module_name (str): The module name of reward.
            class_name (str): The class name of reward.
        """

        return self.workload(RLJobBuilder.REW_ROLE, module_name, class_name)

    def critic(self, module_name, class_name):
        """
        Setup critic.

        Args:
            module_name (str): The module name of actor.
            class_name (str): The class name of actor.
        """

        return self.workload(RLJobBuilder.CRITIC_ROLE, module_name, class_name)

    def validate(self) -> bool:
        if not super(RLJobBuilder, self).validate():
            return False

        if RLJobBuilder.ACTOR_ROLE not in list(self._role_builders.keys()):
            logger.error("'actor' must be configured.")
            return False

        for role, _ in self._role_builders.items():
            if role not in RLJobBuilder.ROLES:
                logger.error(
                    f"{role} is invalid for rl, supported roles "
                    f"are {RLJobBuilder.ROLES}."
                )
                return False

        return True
