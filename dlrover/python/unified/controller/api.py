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
from typing import Dict, List

from dlrover.python.unified.common.actor_base import ActorInfo
from dlrover.python.unified.common.enums import ExecutionResult, MasterStage
from dlrover.python.unified.util.actor_helper import actor_call

MASTER_ACTOR_NAME = "__prime_master__"


@dataclass
class MasterStatus:
    """Status of the master actor."""

    stage: MasterStage
    exit_code: int = 0
    job_restart_count: int = 0


class PrimeMasterApi:
    """Stub for Remote interface for PrimeMaster.

    Note: use async_call inside event loop, otherwise will block the loop.
    """

    ACTOR_NAME: str = MASTER_ACTOR_NAME

    def __init__(self, actor_name: str) -> None:
        """Bind to specified actor name."""
        self.ACTOR_NAME = actor_name

    @actor_call
    @staticmethod
    def get_status() -> MasterStatus:
        """Get the status of the master."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def start() -> None:
        """Start the master."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def restart() -> None:
        """Restart the entire job, mainly used for failover."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def wait() -> None:
        """Wait for the job to finish."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def stop() -> None:
        """Stop the master."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def shutdown() -> None:
        """Force shutdown the master."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def get_actor_info(
        name: str,
    ) -> ActorInfo:
        """Get a actor by name."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def get_workers_by_role(
        role: str, optional: bool = False
    ) -> List[ActorInfo]:
        """Get all actors by role."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def get_all_roles() -> Dict[str, List[ActorInfo]]:
        """Get all roles."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def restart_actors(
        actor_names: List[str],
    ) -> None:
        """Restart the specified actors by names."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def report_actor_restarted(name: str):
        """Report an actor restarted, do failover if needed."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def register_data_queue(name: str, owner_actor: str, size: int):
        """Register a data queue."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def get_data_queue_owner(name: str) -> str:
        """Get the owner actor of a data queue. Waits if not available."""
        raise NotImplementedError("stub")  # pragma: no cover

    @actor_call
    @staticmethod
    def report_execution_result(name: str, result: ExecutionResult):
        """Report the execution result of an actor."""
        raise NotImplementedError("stub")  # pragma: no cover
