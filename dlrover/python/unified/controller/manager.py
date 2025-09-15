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

import asyncio
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import ActorBase, NodeInfo
from dlrover.python.unified.common.constant import MASTER_STATE_KEY_PREFIX
from dlrover.python.unified.common.enums import ExecutionResult, MasterStage
from dlrover.python.unified.controller.events import ControllerEvents
from dlrover.python.unified.controller.extension import ManagerExtension
from dlrover.python.unified.controller.state_backend import MasterStateBackend
from dlrover.python.unified.util.actor_helper import (
    kill_actors,
    restart_actors,
)
from dlrover.python.unified.util.actor_proxy import (
    SELF,
    invoke_actor_t,
    invoke_actors_t,
)

from ..common.config import JobConfig
from .schedule.graph import DLExecutionGraph, DLExecutionVertex
from .schedule.scheduler import Scheduler
from .sync_manager import SyncManager


@dataclass(kw_only=True)
class RuntimeState:
    """
    Dataclass for manager's runtime state. It records all
    the key statement during job runtime. Master(manager) can reload the
    runtime state when the master(manager) actor restart for failover.
    """

    stage: MasterStage = MasterStage.INIT
    exit_code: int = 0
    job_restart_count: int = 0
    node_restart_count: int = 0

    graph: DLExecutionGraph
    removed_nodes: Set[str] = field(default_factory=set)

    extra: Dict[str, Any] = field(default_factory=dict)


class PrimeManager:
    INSTANCE: "PrimeManager"  # Avoid access if possible

    def __init__(self, config: JobConfig) -> None:
        # Note: all fields are readonly default, and state are private mutable

        self.config = config
        self._state_key = MASTER_STATE_KEY_PREFIX + config.job_name

        # Create all components
        self.scheduler: Scheduler = Scheduler(config)
        self.sync_manager = SyncManager()
        self.state_backend = MasterStateBackend.create(
            config.master_state_backend_type,
            config.master_state_backend_config,
        )
        self.ext = ManagerExtension.singleton()

        # Runtime state
        # Note: All states should only mutate by manager itself or methods.
        self.state: RuntimeState = self._load_state() or RuntimeState(
            graph=DLExecutionGraph.create(config.dl_config),
        )
        self._notify_main_loop = asyncio.Semaphore()
        self._stopped_event = asyncio.Event()
        self._task = None

        logger.info(f"PrimeManager initialized with config: {config}")
        PrimeManager.INSTANCE = self  # Singleton instance
        ControllerEvents.inited(config)

    @property
    def graph(self) -> DLExecutionGraph:
        """Get the graph of the job."""
        return self.state.graph

    @property
    def stage(self) -> MasterStage:
        """Get the current stage of the job."""
        return self.state.stage

    def _update_stage(self, stage: MasterStage):
        """Update the stage of the job."""
        old_stage = self.stage
        if old_stage != stage:
            logger.info(f"Updating job stage from {old_stage} to {stage}")
            self.state.stage = stage
            ControllerEvents.stage_updated(old_stage, stage)
            self.save()

    async def prepare(self):
        """Prepare all for the job execution.
        Execute only once, not support failover when failed."""
        with ControllerEvents.creating_pg():
            self.scheduler.allocate_placement_group(self.graph)
        with ControllerEvents.creating_actors():
            await self.scheduler.create_actors(self.graph)
        logger.info("Finished creating actors for the job.")

        # Wait for all nodes to be ready
        await self._setup_actors(self.graph.vertices)
        self._update_stage(MasterStage.READY)

        await self._nodes_check()

    async def _setup_actors(self, actors: List[DLExecutionVertex]):
        """Wait for all actors to be ready."""
        with ControllerEvents.setup_actors():
            res = await invoke_actors_t(ActorBase.setup, actors, SELF)
            res.raise_for_errors()
        logger.info("All actors have completed setup.")

        # update all actor's node info
        node_info_res = await invoke_actors_t(
            ActorBase.get_node_info, actors, SELF
        )
        for actor, node_info in zip(actors, node_info_res.results):
            actor.node_info = node_info

    async def _nodes_check(self):
        """Let sub-masters pre-check nodes.
        This is an optional step, recommended to improve fault tolerance.
        """
        sub_masters = [
            role.sub_master.name
            for role in self.graph.roles.values()
            if role.sub_master is not None
        ]
        if not sub_masters:
            return
        with ControllerEvents.node_check():
            res = await invoke_actors_t(
                ActorBase.check_workers, sub_masters, SELF
            )
            res.raise_for_errors()
        logger.info("Masters checked all workers successfully.")

    async def start(self):
        """Execute the job. Start tracking the job status."""
        assert self.stage == MasterStage.READY, (
            f"Cannot start job in stage {self.stage}. Expected stage is READY."
        )
        with ControllerEvents.starting():
            actors = [actor.name for actor in self.graph.vertices]
            res = await invoke_actors_t(ActorBase.start, actors, SELF)
            res.raise_for_errors()

        logger.info("Job started successfully.")
        self._task = asyncio.create_task(self._main_loop(), name="job_monitor")
        self._update_stage(MasterStage.RUNNING)

    async def _main_loop(self):
        """Monitor the actors' status."""
        while self.stage == MasterStage.RUNNING:
            await self._notify_main_loop.acquire()

            # ignore non-driver roles
            results = [
                role.get_result()
                for role in self.graph.roles.values()
                if role.spec.is_driver
            ]
            # all driver roles finished
            if all(result is None for result in results):
                if any(result == ExecutionResult.FAIL for result in results):
                    self.state.exit_code = 1
                    self.request_stop(
                        "All driver roles finished, but some nodes failed."
                    )
                else:
                    self.request_stop(
                        "All driver roles finished successfully."
                    )
                break

        assert self.stage == MasterStage.STOPPING, (
            f"Job stage should be STOPPING, but got {self.stage}."
        )
        self._do_stop()

    async def restart_actors(self, actors: List[DLExecutionVertex]) -> None:
        """Restart the specified actors."""

        for actor in actors:
            actor.restart_count += 1
            if actor.restart_count > actor.spec.max_restart:
                self.request_stop(
                    f"Actor {actor.name} has exceeded the maximum restart count: {actor.restart_count}."
                )
                return
        for actor in actors:
            actor.restarting = True
        await restart_actors([actor.name for actor in actors])
        await self._setup_actors(actors)
        for actor in actors:
            actor.restarting = False
        logger.info(f"Restarted actors: {[actor.name for actor in actors]}")
        self.save()

    async def deal_with_actor_restarting(self, actor: DLExecutionVertex):
        """
        The core logic of this method should only be executed for actors that
        failed due to exceptions.

        Other actors that are actively or passively restarted for other
        reasons should be filtered out by the following `actor.restarting`
        logic.
        """
        # 1. Ignore some cases
        if actor.restarting:
            return  # Actor is already restarting, no need to handle it again.

        if self.stage != MasterStage.RUNNING:
            logger.info(
                f"Current stage is {self.stage}, skipping failover handling."
            )
            return

        # 2. record failure and relaunch fault node if needed

        assert actor.node_info is not None, (
            "actor.node should be set beforehand."
        )
        if actor.node_info.id in self.state.removed_nodes:
            # caused by node relaunch, reset per-node failure count
            actor.per_node_failure_count = 0
            # This is not a failure
        else:
            actor.inc_failure_count()
        if actor.per_node_failure_count > actor.spec.per_node_max_failure:
            logger.info(
                f"Actor {actor.name} trigger node relaunch for "
                f"exceeded the maximum per-node failure count: {actor.spec.per_node_max_failure}."
            )
            await self._relaunch_fault_nodes([actor.node_info])

        # 3. Do failover

        if actor is actor.role.sub_master:
            await self._setup_actors([actor])
            await invoke_actor_t(ActorBase.recover_running, actor.name, SELF)
            return

        if (
            actor.spec.is_role_level_failover_supported()
            and actor.role.sub_master is not None
        ):
            # call sub master do role level failover
            await invoke_actor_t(
                ActorBase.restart_role_level, actor.role.sub_master.name, SELF
            )
            return
        # call job restart
        await self.restart_job()

    async def _relaunch_fault_nodes(self, nodes: List[NodeInfo]):
        if self.state.node_restart_count >= self.config.node_max_restart:
            logger.fatal(
                f"Node relaunch beyond limit: {self.config.node_max_restart}, stop job directly."
            )
            self.request_stop("node relaunch beyond limit")
            return
        self.state.removed_nodes.update(node.id for node in nodes)
        try:
            relaunched_nodes = await self.ext.relaunch_nodes_impl(nodes)
            self.state.node_restart_count += len(relaunched_nodes)
            logger.info(
                f"Total relaunched nodes: {[node.id for node in relaunched_nodes]}"
            )

            # remove not relaunched nodes from
            for n in nodes:
                if n not in relaunched_nodes:
                    self.state.removed_nodes.remove(n.id)
        except Exception:
            logger.exception(
                "Failed to relaunch nodes due to unexpected error. The following node relaunch may not be executed properly."
            )
            for n in nodes:
                self.state.removed_nodes.remove(n.id)

    async def restart_job(self):
        """Restart the job execution."""
        assert self.stage == MasterStage.RUNNING, (
            f"Cannot restart job in stage {self.stage}. "
            "Expected stage is RUNNING."
        )
        assert self._task is not None
        self.state.job_restart_count += 1
        if self.state.job_restart_count > self.config.job_max_restart:
            self.request_stop(
                f"Job has exceeded the maximum restart count: {self.state.job_restart_count}."
            )
            return
        with ControllerEvents.restarting():
            logger.info("Restarting the job execution.")
            self._task.cancel()
            self._update_stage(MasterStage.READY)
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("Monitor task cancelled, proceeding with restart.")
            logger.info("Restarting all actors...")
            await self.restart_actors(self.graph.vertices)
            logger.info("Restarted actors, re-checking their status.")
            await self._nodes_check()
            await self.start()
            logger.info("Job restarted successfully.")

    def request_stop(self, reason: str):
        """Stop the job execution. And clean up resources."""
        if (
            self.stage == MasterStage.STOPPING
            or self.stage == MasterStage.STOPPED
        ):
            return
        logger.info(f"Requesting to stop the job: {reason}")
        ControllerEvents.stop_requested(reason)
        if self.stage == MasterStage.RUNNING:
            self._update_stage(MasterStage.STOPPING)
            self._notify_main_loop.release()
        else:
            # main loop is not running, do stop directly
            self._do_stop()

    def _do_stop(self):
        with ControllerEvents.stopping():
            logger.info("Terminating all actors...")
            kill_actors([node.name for node in self.graph.vertices])
            self._update_stage(MasterStage.STOPPED)
            self._stopped_event.set()
            logger.info("Job stopped successfully.")

    async def wait(self):
        """Wait for the job to finish."""
        await self._stopped_event.wait()
        assert self.stage == MasterStage.STOPPED

    def save(self):
        """Save the job state to persistent storage."""
        with ControllerEvents.saving():
            try:
                data = pickle.dumps(self.state)
                self.state_backend.set(self._state_key, data)
                logger.info("Save runtime context into state.")
            except Exception:
                logger.exception("Failed to save state")

    def _load_state(self) -> Optional[RuntimeState]:
        if not self.state_backend.exists(self._state_key):
            return None
        try:
            with ControllerEvents.loading_state():
                data = self.state_backend.get(self._state_key)
                assert isinstance(data, bytes), (
                    f"Expected bytes, got {type(data)}"
                )
                data = pickle.loads(data)
                assert isinstance(data, RuntimeState)
                return data
        except Exception:
            logger.exception("Failed to load state")
            return None

    def self_recover(self):
        """Handle failover for the master self."""
        logger.info("Handling failover.")
        if self.stage == MasterStage.RUNNING:
            logger.info("Resuming monitoring the job.")
            self._task = asyncio.create_task(
                self._main_loop(), name="job_monitor"
            )
            # TODO SchedulerManager._pg is not recovered
            ControllerEvents.failover_success()
        else:
            # Don't support failover in other stages, which are little possibility
            logger.warning(f"Job is in stage {self.stage}, terminating.")
            ControllerEvents.failover_stop(self.stage)
            self._do_stop()

    async def deal_with_actor_finished(
        self, actor: DLExecutionVertex, result: ExecutionResult
    ):
        """Handle the actor finished event."""
        logger.info(f"Actor {actor.name} reported result {result}.")
        actor.result = result
        self._notify_main_loop.release()
        # TODO handle Failed case failover
