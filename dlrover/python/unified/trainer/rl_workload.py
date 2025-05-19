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
from abc import ABC

from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.trainer.workload import BaseTaskProcessingWorkload


class BaseRLWorkload(BaseTaskProcessingWorkload, ABC):
    def is_actor_role(self):
        return self._role == RLRoleType.ACTOR.name

    def is_rollout_role(self):
        return self._role == RLRoleType.ROLLOUT.name

    def is_reward_role(self):
        return self._role == RLRoleType.REWARD.name

    def is_ref_role(self):
        return self._role == RLRoleType.REFERENCE.name

    def is_critic_role(self):
        return self._role == RLRoleType.CRITIC.name

    def is_actor_or_rollout_device_collocation(self):
        try:
            if (
                (self.is_actor_role() or self.is_rollout_role())
                and RLRoleType.ACTOR.name in self.get_device_collocation()
                and RLRoleType.ROLLOUT.name in self.get_device_collocation()
            ):
                return True
        except TypeError:
            return False
        return False
