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

from typing import ClassVar, List

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.util.extension_util import (
    Extensible,
    load_entrypoints,
)


class ManagerExtension(Extensible):
    """Extension points for PrimeManager.

    To implement a custom extension:
    1. create a subclass of Extension and override the desired methods.
    2. register the subclass using entrypoints in setup.py:
       entry_points={
           'dlrover.unified.extension': [
               'my_extension = my_module:MyExtension',
           ],
       }
    """

    INSTANCE: ClassVar["ManagerExtension"]

    @staticmethod
    def singleton() -> "ManagerExtension":
        if not hasattr(ManagerExtension, "INSTANCE"):
            load_entrypoints("dlrover.unified.extension")
            ManagerExtension.INSTANCE = ManagerExtension.build_mixed_class()()
        return ManagerExtension.INSTANCE

    @property
    def manager(self):
        """Utility for extension to access PrimeManager singleton"""
        from dlrover.python.unified.controller.manager import PrimeManager

        return PrimeManager.INSTANCE

    # region Extension Points Begin

    async def relaunch_nodes_impl(
        self, nodes: List[NodeInfo]
    ) -> List[NodeInfo]:
        """
        Relaunch the specified nodes.

        For best practice:
        1) do not raise exception(try to catch all the possible exceptions)
        2) use return value(ray nodes) to express how many node relaunched successfully

        Args:
            nodes: The list of ray nodes to relaunch.

        Returns:
            A list of ray nodes which have relaunched successfully.
        """
        return []
