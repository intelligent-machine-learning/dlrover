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

from dataclasses import dataclass
from typing import List

from dlrover.python.unified.common.workload_base import ActorInfo, MasterStage
from dlrover.python.unified.util.actor_proxy import ActorProxy

MASTER_ACTOR_NAME = "__prime_master__"


@dataclass
class MasterStatus:
    """Status of the master actor."""

    stage: MasterStage
    exit_code: int = 0


class PrimeMasterApi(ActorProxy):
    """Stub for Remote interface for PrimeMaster."""

    ACTOR_NAME = MASTER_ACTOR_NAME

    @staticmethod
    def get_status() -> MasterStatus:  # type: ignore[empty-body] # program: no cover
        """Get the status of the master."""
        ...

    @staticmethod
    def start() -> None:  # program: no cover
        """Start the master."""

    @staticmethod
    def wait() -> None:  # program: no cover
        """Wait for the job to finish."""

    @staticmethod
    def stop() -> None:  # program: no cover
        """Stop the master."""

    @staticmethod
    def shutdown() -> None:  # program: no cover
        """Force shutdown the master."""

    @staticmethod
    def get_actor_info(  # type: ignore[empty-body]
        name: str,
    ) -> ActorInfo:  # program: no cover
        """Get a actor by name."""
        ...

    @staticmethod
    def get_workers_by_role(  # type: ignore[empty-body]
        role: str,
    ) -> List[ActorInfo]:  # program: no cover
        """Get all actors by role."""
        ...

    @staticmethod
    def restart_actors(
        actors: List[str],
    ) -> None:  # program: no cover
        """Restart the specified actors."""
        ...
