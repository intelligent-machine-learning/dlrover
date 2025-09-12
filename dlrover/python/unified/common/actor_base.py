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
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import ray.actor

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.enums import ACCELERATOR_TYPE, WorkerStage
from dlrover.python.unified.common.workload_desc import WorkloadDesc
from dlrover.python.unified.util.async_helper import init_main_loop

# Note: All info classes are transfer objects; immutability is preferred.


@dataclass(frozen=True)
class JobInfo:
    """Information about a job. Exposed to workers and sub-masters."""

    name: str
    job_id: str
    user_config: Any
    accelerator_type: ACCELERATOR_TYPE


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class NodeInfo:
    """Information about a node. Exposed to masters."""

    id: str
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    envs: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkerState:
    job_info: JobInfo
    actor_info: ActorInfo
    node_info: NodeInfo
    stage: WorkerStage


class ActorBase:
    """Base class for all actors in the DLRover system."""

    def __init__(self, job_info: JobInfo, actor_info: ActorInfo) -> None:
        """Initialize the actor with node information."""
        self.job_info = job_info
        self.actor_info = actor_info
        self.stage: WorkerStage = WorkerStage.INIT
        init_main_loop()

        # Report restart if this actor was reconstructed.
        if (
            ray.is_initialized()
            and ray.get_runtime_context().was_current_actor_reconstructed
        ):
            try:
                self._report_restart()
            except Exception:
                logger.exception("Unexpected error when reporting restart.")
        self._setup()

    @property
    def name(self) -> str:
        return self.actor_info.name

    def _report_restart(self):
        """Report that the actor has been restarted."""
        from dlrover.python.unified.controller.api import PrimeMasterApi

        PrimeMasterApi.report_actor_restarted(self.actor_info.name)

    def __repr__(self):
        # ActorClass, not instance
        if not hasattr(self, "actor_info"):
            return super().__repr__()
        # We display the actor name in ray logging
        return self.name

    # region Hook methods for subclasses to implement
    def _setup(self):
        """Setup the actor/node.

        This method is called during initialization and should be overridden
        by subclasses to perform any necessary setup before the actor/node
        is ready to run.

        Could be asynchronous, but must keep stage not READY until all setup is done.
        And it must update the stage to READY when setup is complete.
        """
        self._update_stage_force(WorkerStage.READY, WorkerStage.INIT)

    def get_stage(self):
        """Get the stage of the actor."""

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

    # region for sub-master
    def check_workers(self):
        """Check the workers of the master."""
        if self.stage != WorkerStage.READY:
            logger.error(
                f"Only READY stage can perform check_workers. current: {self.stage}"
            )
            return
        pass

    def restart_workers(self):
        """
        Restart workers calling from prime master and executed by sub master.
        """

        raise NotImplementedError(
            "The current sub master does not implement restart_workers."
        )

    # region Misc methods

    def get_node_info(self):
        """Get the current actor's ray node's information."""

        node_id = ray.get_runtime_context().node_id.hex()
        hostname, ip_address = env_utils.get_hostname_and_ip()
        return NodeInfo(
            id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            envs=dict(os.environ),
        )

    # region Helper for subclasses

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
