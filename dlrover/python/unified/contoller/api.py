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

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Protocol

from dlrover.python.unified.common.workload_base import ActorInfo, MasterStage
from dlrover.python.unified.util.actor_helper import ActorProxy

MASTER_ACTOR_ID = "__prime_master__"


@dataclass
class MasterStatus:
    """Status of the master actor."""

    stage: MasterStage


class PrimeMasterRemote(Protocol):
    """Stub for Remote interface for PrimeMaster."""

    @abstractmethod
    def get_status(self) -> MasterStatus:
        """Get the status of the master."""

    @abstractmethod
    def start(self) -> None:
        """Start the master."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the master."""

    @abstractmethod
    def shutdown(self) -> None:
        """Force shutdown the master."""

    @abstractmethod
    def get_actor_info(self, name: str) -> ActorInfo:
        """Get a actor by name."""

    @abstractmethod
    def get_workers_by_role(self, role: str) -> List[ActorInfo]:
        """Get all actors by role."""


PrimeMasterApi: PrimeMasterRemote = ActorProxy.wrap(
    MASTER_ACTOR_ID, cls=PrimeMasterRemote, lazy=True
)
