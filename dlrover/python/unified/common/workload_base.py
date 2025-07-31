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
from typing import Optional

import ray.actor

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_desc import WorkloadDesc
from dlrover.python.unified.util.async_helper import init_main_loop


class MasterStage(str, Enum):
    INIT = "INIT"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class WorkerStage(str, Enum):
    """Stages of a worker actor."""

    # CALL __init__
    INIT = "INIT"
    # _setup
    # _self_check(optional)
    READY = "READY"
    # CALL check_workers(optional)
    # CALL start
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
    user_config: dict


@dataclass
class ActorInfo:
    """Information about a actor. Exposed to workers and sub-masters."""

    name: str
    role: str
    spec: WorkloadDesc
    sub_master: Optional[str] = None

    # common rank information, may be used in rendezvous
    rank: int = 0  # global rank in role
    node_rank: int = 0  # node rank in role
    local_rank: int = 0  # local rank in node


class ActorBase:
    def __init__(self, job_info: JobInfo, actor_info: ActorInfo) -> None:
        """Initialize the actor with node information."""
        self.job_info = job_info
        self.actor_info = actor_info
        self.stage: WorkerStage = WorkerStage.INIT
        init_main_loop()

        # Report restart to sub-master/master if this actor was reconstructed.
        if (
            ray.is_initialized()
            and ray.get_runtime_context().was_current_actor_reconstructed
        ):
            self._report_restart()
        self._setup()

    @property
    def name(self) -> str:
        return self.actor_info.name

    def _report_restart(self):
        """Report that the actor has been restarted."""
        from dlrover.python.unified.controller.api import PrimeMasterApi
        from dlrover.python.unified.util.actor_proxy import invoke_actor_t

        master = self.actor_info.sub_master or PrimeMasterApi.ACTOR_NAME
        invoke_actor_t(
            PrimeMasterApi.report_actor_restarted,
            master,
            name=self.actor_info.name,
        ).wait()

    # Hook methods for subclasses to implement
    def _setup(self):
        """Setup the actor/node.

        This method is called during initialization and should be overridden
        by subclasses to perform any necessary setup before the actor/node
        is ready to run.

        Could be asynchronous, but must keep stage not READY until all setup is done.
        And it must update the stage to READY when setup is complete.
        """
        self._update_stage_force(WorkerStage.READY, WorkerStage.INIT)

    def status(self):
        """Get the state of the actor/node."""
        return self.stage

    def start(self):
        """Start the actor/node. If already started, do nothing.

        This method should be overridden by subMaster or trainer,
        depending on the usage pattern.

        Noticed:
        1. The worker stage must be 'RUNNING' after the method invocation.
        2. Main processing need to be defined in an async thread under the
           worker actor, and ensure that stage updated to FINISHED or FAILED when done.
        """

    def shutdown(self):
        """Self-kill the actor/node."""
        ray.actor.exit_actor()  # As ray.kill don't execute callback.

    # for sub-master
    def check_workers(self):
        """Check the workers of the master."""
        if self.stage != WorkerStage.READY:
            logger.error(
                f"Only READY stage can perform check_workers. current: {self.stage}"
            )
            return
        pass

    def report_actor_restarted(self, name: str):
        """Report that the actor has been restarted."""

        from dlrover.python.unified.controller.api import PrimeMasterApi

        # default delegate to master
        PrimeMasterApi.report_actor_restarted(name)

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
