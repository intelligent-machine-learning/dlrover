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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import ActorBase, NodeInfo
from dlrover.python.unified.common.constant import (
    MASTER_STATE_KEY_PREFIX,
    RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME,
    InternalDLWorkloadRole,
)
from dlrover.python.unified.common.enums import ExecutionResult, MasterStage
from dlrover.python.unified.controller.events import ControllerEvents
from dlrover.python.unified.controller.extension import ManagerExtension
from dlrover.python.unified.controller.state_backend import MasterStateBackend
from dlrover.python.unified.util.actor_helper import (
    SELF,
    invoke_actor,
    invoke_actors,
    kill_actors,
    restart_actors,
    wait_ray_node_remove,
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
    # lag for failover status, the presence of a corresponding key-value pair
    # indicates that failover is in progress.
    # format:
    # key: role name, is 'GLOBAL' if job level
    # value: timestamp
    failover_stage: Dict[str, int] = field(default_factory=dict)

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
            res = await invoke_actors(ActorBase.setup, actors, SELF)
            res.raise_for_errors()
        logger.info("All actors have completed setup.")

        # update all actor's node info
        node_info_res = await invoke_actors(
            ActorBase.get_node_info, actors, SELF
        )
        for actor, node_info in zip(actors, node_info_res.results):
            actor.node_info = node_info
            actor.is_ready.set()

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
            res = await invoke_actors(
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
            res = await invoke_actors(ActorBase.start, actors, SELF)
            res.raise_for_errors()

        logger.info("Job started successfully.")
        self._task = asyncio.create_task(self._main_loop(), name="job_monitor")
        self._update_stage(MasterStage.RUNNING)

    async def _main_loop(self):
        """Monitor the actors' status."""
        while self.stage == MasterStage.RUNNING:
            await self._notify_main_loop.acquire()

            any_failure = [
                role.has_any_failure() for role in self.graph.roles.values()
            ]
            if any(any_failure):
                if self.config.failover_trigger_strategy == 2:
                    await self._process_failover(reason="got worker failure")
                logger.info(
                    "Failure detected, but since the failover-trigger-strategy is set to 0, failover will not be executed for now."
                )

            # ignore non-driver roles
            results = [
                role.get_result()
                for role in self.graph.roles.values()
                if role.spec.is_driver
            ]
            # all driver roles finished
            if all(result is not None for result in results):
                if any(result == ExecutionResult.FAIL for result in results):
                    await self._process_failover(
                        reason="got failure result on finished"
                    )
                else:
                    self.request_stop(
                        "All driver roles finished successfully.", code=0
                    )
                break

            await asyncio.sleep(1)

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
        logger.info(f"Actor {actor.name} is restarting.")
        actor.is_ready.clear()  # reset to unready after restart whatever the reason is

        # Ignore some cases
        if actor.restarting:
            return  # Actor is already restarting, no need to handle it again.

        if self.stage != MasterStage.RUNNING:
            logger.info(
                f"Current stage is {self.stage}, skipping failover handling."
            )
            return

        # record failure

        assert actor.node_info is not None, (
            "actor.node should be set beforehand."
        )
        if actor.node_info.id in self.state.removed_nodes:
            # caused by node relaunch, reset per-node failure count
            actor.per_node_failure_count = 0
            # This is not a failure
        else:
            actor.inc_failure_count()

        # Do failover
        asyncio.create_task(self._do_failover(actor))

    async def _do_failover(self, actor: DLExecutionVertex):
        """Handle failover for the given actor."""
        # relaunch fault node if needed
        if actor.per_node_failure_count > actor.spec.per_node_max_failure:
            assert actor.node_info is not None
            logger.info(
                f"Actor {actor.name} trigger node relaunch for "
                f"exceeded the maximum per-node failure count: {actor.spec.per_node_max_failure}."
            )
            await self._relaunch_fault_nodes([actor.node_info])

        # if the actor is sub-master, recover it directly
        if actor is actor.role.sub_master:
            await self._setup_actors([actor])
            await invoke_actor(ActorBase.recover_running, actor.name, SELF)
            return

        # Let sub-master handle worker failover first
        if actor.role.sub_master is not None:
            await actor.role.sub_master.is_ready.wait()
            handled = await invoke_actor(
                ActorBase.handle_worker_failover,
                actor.role.sub_master.name,
                SELF,
                actor.name,
            )
            # If the sub-master handled the failover, we're done
            if handled:
                return
        # fallback to job restart to handle the failover
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
            # unset non-relaunched nodes
            for n in nodes:
                if n not in relaunched_nodes:
                    self.state.removed_nodes.remove(n.id)
            if not relaunched_nodes:
                logger.info("No nodes were relaunched.")
                return
            # Ensure the nodes are removed from Ray cluster, avoid exceptions during restart processing.
            await asyncio.wait_for(
                wait_ray_node_remove([n.id for n in relaunched_nodes]),
                timeout=(
                    len(relaunched_nodes) * RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME
                ),
            )
            logger.info(
                f"Relaunched nodes: {[node.id for node in relaunched_nodes]}"
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout waiting for nodes to be removed from Ray cluster, may cause inconsistency."
            )
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

    def request_stop(self, reason: str, code: int = 1):
        """Stop the job execution. And clean up resources."""
        if (
            self.stage == MasterStage.STOPPING
            or self.stage == MasterStage.STOPPED
        ):
            return
        logger.info(f"Requesting to stop the job: {reason}(code={code})")
        ControllerEvents.stop_requested(reason, code)

        self.state.exit_code = code
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

            # if a master failover and a failover of any role overlap,
            # we currently perform a global restart again for safety.
            if self.is_failover_stage():
                self._clear_failover_stage()
                asyncio.create_task(
                    self._process_failover(reason="failover overlapping")
                )

            # TODO SchedulerManager._pg is not recovered
            ControllerEvents.failover_success()
        else:
            # Don't support failover in other stages, which are little possibility
            logger.warning(f"Job is in stage {self.stage}, terminating.")
            ControllerEvents.failover_stop(self.stage)
            self._clear_failover_stage()
            self._do_stop()

    async def deal_with_actor_finished(
        self, actor: DLExecutionVertex, result: ExecutionResult
    ):
        """Handle the actor finished event."""
        logger.info(f"Actor {actor.name} reported result {result}.")
        actor.result = result
        self._notify_main_loop.release()

    def _set_failover_stage(self, role_name):
        if role_name not in self.state.failover_stage:
            self.state.failover_stage[role_name] = int(time.time())
            logger.debug(f"Setting failover stage: {role_name}")

    def _clear_failover_stage(self, role_name=""):
        if not role_name:
            self.state.failover_stage.clear()
            logger.debug("Clear all failover stage.")
        else:
            self.state.failover_stage.pop(role_name)
            logger.debug(f"Clear failover stage: {role_name}")

    def is_failover_stage(self, role_name=""):
        if not role_name:
            # is any role in failover stage
            return len(self.state.failover_stage) == 0
        else:
            if (
                InternalDLWorkloadRole.GLOBAL_ROLE in self.state.failover_stage
                or role_name in self.state.failover_stage
            ):
                return True
            return False

    async def _process_failover(
        self,
        role_name=InternalDLWorkloadRole.GLOBAL_ROLE,
        reason="unknown reason",
    ):
        if self.is_failover_stage(role_name):
            logger.info(
                "Failover is already in progress, skip the following process."
            )
        else:
            self._set_failover_stage(role_name)

            if self.config.failover_exec_strategy == 1:
                # trigger job failover
                logger.info(f"Trigger job failover, reason: {reason}.")
                await self.restart_job()
            elif self.config.failover_exec_strategy == 2:
                # TODO: implement by role level failover
                logger.info(
                    "Role level failover is not supported yet, do job failover instead, "
                    f"reason: {reason}."
                )
                await self.restart_job()
            else:
                logger.info(
                    "Skip failover for strategy(failover_strategy_when_failed) "
                    f"is: {self.config.failover_exec_strategy}, "
                    f"reason: {reason}."
                )
                # stop job
                self.request_stop(
                    "All driver roles finished, but some workers failed."
                )

            self._clear_failover_stage(role_name)
