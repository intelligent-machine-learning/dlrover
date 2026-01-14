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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import (
    ActorBase,
    NodeInfo,
    ExecutionResult,
    DiagnosticInfo,
)
from dlrover.python.unified.common.constant import (
    MASTER_STATE_KEY_PREFIX,
    RAY_NODE_RELAUNCH_WAIT_TIME,
    InternalDLWorkloadRole,
)
from dlrover.python.unified.common.enums import (
    ExecutionResultType,
    MasterStage,
)
from dlrover.python.unified.controller.events import ControllerEvents
from dlrover.python.unified.controller.extension import ManagerExtension
from dlrover.python.unified.controller.state_backend import MasterStateBackend
from dlrover.python.unified.util.actor_helper import (
    SELF,
    invoke_actor,
    invoke_actors,
    kill_actors,
    restart_actors,
)

from ..common.config import JobConfig
from .schedule.graph import (
    DLExecutionGraph,
    DLExecutionVertex,
    DLExecutionWorkerVertex,
)
from .schedule.scheduler import Scheduler
from .sync_manager import SyncManager
from ..common.workload_desc import NodeGroupFailoverDesc
from ..util.node_helper import wait_ray_node_relaunching, get_node_group


@dataclass(kw_only=True)
class RuntimeState:
    """
    Dataclass for manager's runtime state. It records all
    the key statement during job runtime. Master(manager) can reload the
    runtime state when the master(manager) actor restart for failover.
    """

    stage: MasterStage = MasterStage.INIT
    # flag for failover status, the presence of a corresponding key-value pair
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
        self._state_lock = threading.Lock()
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

        with self._state_lock:
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

        logger.info("Start main task loop...")
        while self._in_running_stage():
            try:
                await self._notify_main_loop.acquire()

                any_failure = [
                    role.has_any_failure()
                    for role in self.graph.roles.values()
                ]
                if any(any_failure):
                    if self.config.failover_trigger_strategy == 2:
                        logger.info(
                            "Failure detected, process failover right now due "
                            "to failover-trigger-strategy is set to 2."
                        )
                        await self._process_failover(
                            reason="got worker failure"
                        )
                    elif self.config.failover_trigger_strategy == 1:
                        logger.info(
                            "Failure detected, but since the failover-trigger-strategy "
                            "is set to 1, failover will not be executed for now."
                        )
                    else:
                        logger.info(
                            "Failure detected, but since the failover-trigger-strategy "
                            "is set to 0, skip failover."
                        )

                # ignore non-driver roles
                results = [
                    role.get_result()
                    for role in self.graph.roles.values()
                    if role.spec.is_driver
                ]

                # all driver roles finished
                if all(result is not None for result in results):
                    if any(
                        result == ExecutionResultType.FAIL
                        for result in results
                    ):
                        await self._process_failover(
                            reason="got failure result on finished"
                        )
                    else:
                        self.request_stop(
                            "All driver roles finished successfully.", code=0
                        )
                    break

                await asyncio.sleep(1)
            except Exception:
                logger.exception(
                    "Unexpected exception occurred in main task loop."
                )
                self.request_stop(
                    "Unexpected exception occurred in main task loop."
                )
                return

        if self.stage == MasterStage.STOPPING:
            self._do_stop()

        logger.info(f"Main task loop exited with stage: {self.stage}")

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
            actor.result = None
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

        # failover can only be executed in serial
        if not self._in_running_stage() or self.is_failover_stage(
            actor.role.name
        ):
            logger.info(
                f"Current stage is {self.stage}(is failover: {self.is_failover_stage(actor.role.name)}), "
                f"skipping failover handling by actor {actor.name}."
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
            logger.info(
                f"Actor is restarted due to node-relaunch: {actor.node_info.id}, "
                f"skipping failover handling by actor {actor.name}."
            )
            return
        else:
            actor.inc_failure_count()

        # Do failover
        asyncio.create_task(self._do_failover(actor))

    async def _do_failover(self, actor: DLExecutionVertex):
        """Handle failover for the given actor."""

        role_name = actor.role.name
        self._set_failover_stage(role_name)
        upgrade_to_global_failover = False

        # relaunch fault node if needed
        if actor.per_node_failure_count > actor.spec.per_node_max_failure:
            assert actor.node_info is not None
            logger.info(
                f"Actor {actor.name} trigger node relaunch for "
                f"exceeded the maximum per-node failure count: {actor.spec.per_node_max_failure}."
            )

            # upgrade to global failover if there is node relaunching process
            upgrade_to_global_failover = True
            self._set_failover_stage(InternalDLWorkloadRole.GLOBAL_ROLE)

            # do node relaunch
            await self._relaunch_single_node_by_actor(actor)

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

        if upgrade_to_global_failover:
            self._clear_failover_stage(InternalDLWorkloadRole.GLOBAL_ROLE)
        self._clear_failover_stage(role_name)

    async def _relaunch_nodes(
        self, nodes: List[NodeInfo], timeout=RAY_NODE_RELAUNCH_WAIT_TIME
    ):
        """
        Relaunch ray nodes.
        """

        if self.state.node_restart_count >= self.config.node_max_restart:
            logger.fatal(
                f"Node relaunch beyond limit: {self.config.node_max_restart}, stop job directly."
            )
            self.request_stop("node relaunch beyond limit")
            return

        for node in nodes:
            self.state.removed_nodes.add(node.id)

        try:
            total_nodes_size = len(ray.nodes())

            async def _relaunch_and_wait():
                relaunched_nodes_by_ext = await self.ext.relaunch_nodes_impl(
                    nodes
                )
                self.state.node_restart_count += len(relaunched_nodes_by_ext)

                # unset non-relaunched nodes
                for target_node in nodes:
                    if target_node not in relaunched_nodes_by_ext:
                        self.state.removed_nodes.discard(target_node.id)
                if not relaunched_nodes_by_ext:
                    logger.info("No nodes were relaunched.")
                    return relaunched_nodes_by_ext

                await wait_ray_node_relaunching(
                    [n.id for n in relaunched_nodes_by_ext], total_nodes_size
                )
                return relaunched_nodes_by_ext

            relaunched_nodes = await asyncio.wait_for(
                _relaunch_and_wait(), timeout=timeout
            )
            logger.info(
                f"Relaunched nodes: {[node.id for node in relaunched_nodes]}"
            )
        except asyncio.TimeoutError as ate:
            raise ate
        except Exception as e:
            for node in nodes:
                self.state.removed_nodes.discard(node.id)
            raise e

    async def _relaunch_single_node_by_actor(
        self, root_cause_actor: DLExecutionVertex
    ):
        """
        For single node relaunch.
        This method, because it operates on only a single node, can determine
        the relaunch status of the target node by controlling variables,
        thereby supporting the advanced implementation of node group relaunch.
        """

        node: Optional[NodeInfo] = root_cause_actor.node_info
        node_group_failover_info: NodeGroupFailoverDesc = (
            root_cause_actor.spec.node_group_failover
        )
        if not node:
            return

        logger.info(
            f"Relaunch node {node.id} "
            f"with node_group_failover: {node_group_failover_info} "
            f"due to {root_cause_actor.name}."
        )

        if (
            node_group_failover_info.enabled
            and node_group_failover_info.group_label_key
        ):
            node_group_failover_timeout = node_group_failover_info.timeout
            timeout = min(
                RAY_NODE_RELAUNCH_WAIT_TIME, node_group_failover_timeout
            )
        else:
            timeout = RAY_NODE_RELAUNCH_WAIT_TIME

        try:
            await self._relaunch_nodes([node], timeout)

            # reset root cause actors failure info
            root_cause_actor.reset_per_node_failure_count()
        except asyncio.TimeoutError:
            if node_group_failover_info:
                group_nodes = get_node_group(
                    node, node_group_failover_info.group_label_key
                )
                await self._relaunch_node_group(node, group_nodes)
        except Exception:
            logger.exception(
                "Failed to relaunch node due to unexpected error. The following node relaunch may not be executed properly."
            )
            # TODO: may need to request stop due to configuration

    async def _relaunch_node_group(
        self, root_node: NodeInfo, node_group: List[NodeInfo]
    ):
        """
        For node group relaunch.
        This method, for batch node relaunch, due to Ray's mechanism,
        cannot determine the relaunch result of a specific node and can only
        obtain the overall relaunch result.
        """

        logger.info(f"Relaunch node-group {node_group} due to: {root_node}.")

        try:
            return await self._relaunch_nodes(
                node_group, timeout=RAY_NODE_RELAUNCH_WAIT_TIME * 2
            )
        except Exception:
            logger.exception(
                "Failed to relaunch node-group due to unexpected error. The following node relaunch may not be executed properly."
            )
            # TODO: may need to request stop due to configuration

    async def restart_job(
        self, with_node_relaunch: Optional[DLExecutionVertex] = None
    ):
        """
        Restart the job execution.

        Args:
            with_node_relaunch: With node relaunch info,
                format: target node/node group label/node group failover timeout
        """

        assert self._in_running_stage(), (
            f"Cannot restart job in stage {self.stage}. "
            "Expected stage is RUNNING."
        )
        assert self._task is not None

        # do node relaunch before restarting
        if with_node_relaunch:
            await self._relaunch_single_node_by_actor(with_node_relaunch)

        # restarting
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
            except RuntimeError:
                logger.warning(
                    f"Unexpected runtime error when await task: {type(self._task)}"
                )

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
        if self._in_running_stage():
            self._update_stage(MasterStage.STOPPING)
            self._notify_main_loop.release()
        else:
            # main loop is not running, do stop directly
            self._do_stop()

    def _in_running_stage(self):
        """Including running stages."""

        return self.stage == MasterStage.RUNNING

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
        if self._in_running_stage():
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

    async def deal_with_actor_diagnostic_report(
        self, actor: DLExecutionVertex, diagnostic: DiagnosticInfo
    ):
        """Handle the actor diagnostic info reporting."""

        logger.info(f"Actor {actor.name} reported diagnostic {diagnostic}.")
        actor.diagnostic = diagnostic

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

    def _record_failure(self, role_name):
        """
        Update failure info into context.

        For now:
        The number of update failures is not simply incremented by one.
        Instead, it is determined by evaluating the execution results of all
        actors associated with the current role, which are then sorted and
        selected based on result priority and timestamp. The failure record of
        the specified actor is then updated.

        TODO: may need more diagnosis operations here
        """

        target_instances: List[DLExecutionWorkerVertex] = []
        if role_name == InternalDLWorkloadRole.GLOBAL_ROLE:
            for role in self.graph.roles.values():
                target_instances.extend(role.instances)
        else:
            target_instances.extend(self.graph.roles[role_name].instances)

        if any(
            instance.is_failure_responsibility()
            for instance in target_instances
        ):
            # update specified if there is a known cause
            for instance in target_instances:
                if instance.is_failure_responsibility():
                    instance.inc_failure_count()
        else:
            # otherwise for all unknown or be affected: pick up the 1st failure as root
            failed_instances = [
                instance
                for instance in target_instances
                if instance.is_failure()
            ]
            failed_instances.sort(
                key=lambda x: x.result.timestamp if x.result else int("inf")
            )
            if failed_instances:
                failed_instances[0].inc_failure_count()

    def _get_node_relaunch_demand(self, role_name):
        if role_name == InternalDLWorkloadRole.GLOBAL_ROLE:
            for role in self.graph.roles.values():
                with_node_relaunch_demand_actor = (
                    role.get_node_relaunch_demand_actor()
                )
                if with_node_relaunch_demand_actor:
                    break
        else:
            with_node_relaunch_demand_actor = self.graph.roles[
                role_name
            ].get_node_relaunch_demand_actor()

        if with_node_relaunch_demand_actor:
            logger.info(
                f"Got node relaunch demand actor when processing failover: {with_node_relaunch_demand_actor.name}"
            )

        return with_node_relaunch_demand_actor

    async def _process_failover(
        self,
        role_name=InternalDLWorkloadRole.GLOBAL_ROLE,
        reason="unknown reason",
    ):
        if self.is_failover_stage(role_name):
            logger.info(
                "Failover is already in progress, skip the following process."
            )
            return

        logger.info(f"Processing failover for {role_name}")
        self._set_failover_stage(role_name)

        # update failure info into context
        self._record_failure(role_name)

        # trigger node relaunch according to the failure info
        node_relaunch_demand_actor: Optional[DLExecutionVertex] = (
            self._get_node_relaunch_demand(role_name)
        )

        if self.config.failover_exec_strategy == 1:
            # trigger job failover
            logger.info(f"Trigger job failover, reason: {reason}.")
            await self.restart_job(
                with_node_relaunch=node_relaunch_demand_actor
            )
        elif self.config.failover_exec_strategy == 2:
            # TODO: implement by role level failover
            logger.info(
                "Role level failover is not supported yet, do job failover instead, "
                f"reason: {reason}."
            )
            await self.restart_job(
                with_node_relaunch=node_relaunch_demand_actor
            )
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
