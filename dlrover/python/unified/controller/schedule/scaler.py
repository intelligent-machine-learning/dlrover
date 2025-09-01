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
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.controller.auto_registry import (
    extensible,
    AutoExtensionRegistry,
)


def get_scaler(config):
    extension_cls = AutoExtensionRegistry.get_extension_class_by_interface(
        BaseRayNodeScaler.__qualname__
    )
    if extension_cls:
        logger.info(f"Using extension scaler: {extension_cls.__qualname__}")
        return extension_cls(config)

    logger.info(f"Using default scaler: {DefaultRayNodeScaler.__qualname__}")
    return DefaultRayNodeScaler(config)


@extensible()
class BaseRayNodeScaler(ABC):
    """
    The abstraction of scaler.
    Scaler is used to scale ray node during runtime.
    """

    def __init__(self, config: JobConfig):
        self.config = config

    @abstractmethod
    def relaunch(self, target_nodes: List[NodeInfo]) -> bool:
        """
        Relaunch the specified ray nodes.

        Args:
            target_nodes: Specified nodes to relaunch.
        """

    @abstractmethod
    def scale_up(
        self, count: int = 1, resource: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Scale up specified number of ray nodes.

        Args:
            count: Number of ray nodes to scale.
            resource: Specified resource to scale. Default is None(with same
                resource of current node).
        """

    @abstractmethod
    def scale_down(self, target_nodes: List[str]) -> bool:
        """
        Scale down the specified ray nodes.

        Args:
            target_nodes: Specified nodes to scale down. The specified element
            can be freely implemented by subclasses and may be a node ID,
            node hostname, IP address, or the name of the corresponding pod.
        """


class DefaultRayNodeScaler(BaseRayNodeScaler):
    def relaunch(self, target_nodes: List[NodeInfo]) -> bool:
        logger.warning("Default scaler does not support node relaunch.")
        return False

    def scale_up(
        self, count: int = 1, resource: Optional[Dict[str, Any]] = None
    ) -> bool:
        logger.warning("Default scaler does not support node scaling up.")
        return False

    def scale_down(self, target_nodes: List[str]) -> bool:
        logger.warning("Default scaler does not support node scaling down.")
        return False
