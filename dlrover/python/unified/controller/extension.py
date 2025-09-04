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

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.util.extension_util import Extensible


class Extension(Extensible):
    INSTANCE: "Extension"

    @staticmethod
    def singleton() -> "Extension":
        if not Extension.INSTANCE:
            Extension.INSTANCE = Extension.build_mixed_class()()
        return Extension.INSTANCE

    async def relaunch_nodes_impl(self, nodes: List[NodeInfo]):
        """Relaunch the specified nodes.
        @param nodes: The list of ray node IDs to relaunch.
        """
        raise NotImplementedError("Relaunch is not implemented")
