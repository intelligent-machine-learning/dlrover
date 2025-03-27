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
from abc import ABC, abstractmethod
from typing import Dict, List

from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.rl.common.enums import RLRoleType


class BaseTrainer(ABC):
    def __init__(
        self,
        actor_handles: Dict[RLRoleType, List[ActorHandle]],
        config: DictConfig,
    ):
        self._actor_handles = actor_handles
        self._config = config

    @property
    def actor_handles(self) -> Dict[RLRoleType, List[ActorHandle]]:
        return self._actor_handles

    @property
    def config(self) -> DictConfig:
        return self._config

    def get_actor_handles(self, role_type: RLRoleType) -> List[ActorHandle]:
        return self._actor_handles[role_type]

    @abstractmethod
    def init(self):
        """
        Requires user implementation: for initializing all workers,
        such as loading the dataloader, loading the model, etc.
        """

    @abstractmethod
    def fit(self):
        """
        Requires user implementation: the core training logic.
        """
