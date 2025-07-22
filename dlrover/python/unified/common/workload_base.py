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
from enum import Enum
from typing import Optional, Union

import ray.actor
from omegaconf import DictConfig

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_desc import WorkloadDesc
from dlrover.python.unified.util.test_hooks import init_coverage

init_coverage()  # support coverage for workers actor


class MasterStage(str, Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class WorkerStage(str, Enum):
    """Stages of a worker actor."""

    INIT = "INIT"
    PENDING = "PENDING"  # Checking
    READY = "READY"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"

    def is_terminal(self) -> bool:
        """Check if the stage is terminal."""
        return self in {WorkerStage.FINISHED, WorkerStage.FAILED}


@dataclass
class JobInfo:
    """Information about a job. Exposed to workers and sub-masters."""

    name: str
    job_id: str
    user_config: Union[dict, DictConfig]


@dataclass
class ActorInfo:
    """Information about a actor. Exposed to workers and sub-masters."""

    name: str
    role: str
    spec: WorkloadDesc

    # common rank information, may be used in rendezvous
    rank: int = 0  # global rank in role
    node_rank: int = 0  # node rank in role
    local_rank: int = 0  # local rank in node


class ActorBase:
    def __init__(self, job_info: JobInfo, actor_info: ActorInfo) -> None:
        """Initialize the actor with node information."""
        init_coverage()
        self.job_info = job_info
        self.actor_info = actor_info
        self.stage: WorkerStage = WorkerStage.INIT
        self._setup()

    # Hook methods for subclasses to implement
    def _setup(self):
        """Setup the actor/node."""
        pass

    def status(self):
        """Get the state of the actor/node."""
        return self.stage

    def self_check(self):
        """Check the actor/node itself."""
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return  # already in the expected stage
        logger.info(f"[{self.actor_info.name}] Running self check.")

    def start(self):
        """Start the actor/node. If already started, do nothing.

        This method should be overridden by subMaster or trainer,
        depending on the usage pattern.

        Noticed:
        1. The worker stage must be 'RUNNING' after the method invocation.
        2. Main processing need to be defined in a async thread under the
           worker actor.
        """

    def shutdown(self):
        """Self-kill the actor/node."""
        ray.actor.exit_actor()  # As ray.kill don't execute callback.

    # for sub-master
    def check_workers(self):
        """Check the workers of the master."""
        pass

    # Helper methods for subclasses to use

    def _update_stage_force(
        self, stage: WorkerStage, expected: Optional[WorkerStage] = None
    ):
        """Update the stage of the actor/node."""
        if expected is not None and self.stage != expected:
            raise RuntimeError(
                f"Cannot update stage from {self.stage} to {stage}, "
                f"expected {expected}."
            )
        self.stage = stage
        logger.info(
            f"Actor {self.actor_info.name} updated to stage: {self.stage}"
        )

    def _update_stage_if(self, stage: WorkerStage, expected: WorkerStage):
        """Update the stage of the actor/node
        if the current stage matches the expected stage.
        """
        if self.stage != expected:
            logger.warning(
                f"Actor {self.actor_info.name} is not in expected stage: "
                f"{expected}, current stage: {self.stage}"
            )
            return False  # not in the expected stage
        self.stage = stage
        logger.info(
            f"Actor {self.actor_info.name} updated to stage: {self.stage}"
        )
        return True
