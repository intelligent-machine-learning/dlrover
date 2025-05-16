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
from typing import List

from ray.actor import ActorHandle

from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.trainer.trainer import BaseTrainer


class BaseRLTrainer(BaseTrainer, ABC):
    @property
    def actors(self) -> List[ActorHandle]:
        """Get all actors' actor handle."""
        return self.get_actor_handles(RLRoleType.ACTOR.name)

    @property
    def actor_resource(self):
        """Get resource used(occupied) for ACTOR."""
        return self.get_workload_resource(RLRoleType.ACTOR.name)

    @property
    def references(self) -> List[ActorHandle]:
        """Get all references' actor handle."""
        return self.get_actor_handles(RLRoleType.REFERENCE.name)

    @property
    def reference_resource(self):
        """Get resource used(occupied) for REFERENCE."""
        return self.get_workload_resource(RLRoleType.REFERENCE.name)

    @property
    def rollouts(self) -> List[ActorHandle]:
        """Get all rollouts' actor handle."""
        return self.get_actor_handles(RLRoleType.ROLLOUT.name)

    @property
    def rollout_resource(self):
        """Get resource used(occupied) for ROLLOUT."""
        return self.get_workload_resource(RLRoleType.ROLLOUT.name)

    @property
    def rewards(self) -> List[ActorHandle]:
        """Get all rewards' actor handle."""
        return self.get_actor_handles(RLRoleType.REWARD.name)

    @property
    def reward_resource(self):
        """Get resource used(occupied) for REWARD."""
        return self.get_workload_resource(RLRoleType.REWARD.name)

    @property
    def critics(self) -> List[ActorHandle]:
        """Get all critics' actor handle."""
        return self.get_actor_handles(RLRoleType.CRITIC.name)

    @property
    def critic_resource(self):
        """Get resource used(occupied) for CRITIC."""
        return self.get_workload_resource(RLRoleType.CRITIC.name)
