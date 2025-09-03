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
from typing import List, Dict, Any, Tuple

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import WorkerStage, NodeInfo
from dlrover.python.unified.common.enums import MasterStage
from dlrover.python.unified.util.actor_helper import (
    kill_actors,
    restart_actors,
)
from dlrover.python.unified.util.actor_proxy import (
    invoke_actors_t,
    invoke_actor_t,
)
from .schedule.scaler import BaseRayNodeScaler, get_scaler
from .state_backend import MasterStateBackendFactory

from ..common.config import JobConfig
from . import remote_call
from .api import MasterStatus
from .schedule.graph import DLExecutionGraph, DLExecutionVertex
from .schedule.scheduler import Scheduler
from .sync_manager import SyncManager
from ..common.constant import (
    RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME,
    MasterConstant,
)


@dataclass(kw_only=True)
class ManagerRuntimeContext(object):
    """
    This context is used for manager's runtime stateful using. It records all
    the key statement during job runtime. Master(manager) can reload the
    runtime context when the master(manager) actor restart for failover.
    """

    graph: DLExecutionGraph
    stage: MasterStage
    status: MasterStatus

    node_total_count: int
    node_restart_count: int = 0

    extra: Dict[str, Any] = field(default_factory=dict)

    def serialize(self):
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data):
        deserialized = pickle.loads(data)
        assert isinstance(deserialized, cls)
        return deserialized


class PrimeManager:
    INSTANCE: "PrimeManager"

    def __init__(self, config: JobConfig) -> None:
        self.config = config
        self._state_backend = MasterStateBackendFactory.get_state_backend(
            config.master_state_backend_type
        )

        # Create all components
        self.scheduler: Scheduler = Scheduler(config)
        self.scaler: BaseRayNodeScaler = get_scaler(config)
        self.sync_manager = SyncManager()

        # Runtime state
        is_init, self._context = self._init_or_reload()
        if is_init:
            self.save()
        self._stopped_event = asyncio.Event()

        logger.info(f"PrimeManager initialized with config: {config}")
        self.INSTANCE = self  # Singleton instance
        self._task = None

    @property
    def context(self) -> ManagerRuntimeContext:
        """Get the runtime context."""
        return self._context

    @property
    def graph(self) -> DLExecutionGraph:
        """Get the graph of the job."""
        return self.context.graph

    @property
    def stage(self) -> MasterStage:
        """Get the current stage of the job."""
        return self.context.stage

    @property
    def status(self) -> MasterStatus:
        """Get the current status of the job."""
        return self.context.status

    def _update_stage(self, stage: MasterStage):
        """Update the stage of the job."""
        if self.stage != stage:
            logger.info(f"Updating job stage from {self.stage} to {stage}")
            self.context.stage = stage
            self.context.status.stage = stage
            self.save()

    async def prepare(self):
        """Prepare all for the job execution.
        Execute only once, not support failover when failed."""
        self.scheduler.allocate_placement_group(self.graph)
        await self.scheduler.create_actors(self.graph)
        logger.info("Finished creating actors for the job.")

        # Wait for all nodes to be ready
        await self._wait_ready(self.graph.vertices)
        self._update_stage(MasterStage.READY)

        await self._nodes_check()

    async def _wait_ready(self, actors: List[DLExecutionVertex]):
        """Wait for all actors to be ready."""
        while True:
            res = await invoke_actors_t(remote_call.get_stage, actors)
            not_ready = {
                actor.name: status
                for actor, status in zip(actors, res.results)
                if status != "READY"
            }
            if len(not_ready) == 0:
                logger.info("All nodes are ready.")
                break
            logger.warning(
                f"Waiting for {len(not_ready)} nodes to be ready: {not_ready}"
            )
            await asyncio.sleep(5)

        # update all actor's node info
        node_info_res = await invoke_actors_t(
            remote_call.get_node_info, actors
        )
        for actor, node_info in zip(actors, node_info_res.results):
            actor.set_node(node_info)

    async def _nodes_check(self):
        # let sub-masters pre-check nodes
        sub_masters = [
            role.sub_master.name
            for role in self.graph.roles.values()
            if role.sub_master is not None
        ]
        if sub_masters:
            res = await invoke_actors_t(remote_call.check_workers, sub_masters)
            res.raise_for_errors()
            logger.info("Masters checked all workers successfully.")

    async def start(self):
        """Execute the job. Start tracking the job status."""
        if self.stage != MasterStage.READY:
            raise RuntimeError(
                f"Cannot start job in stage {self.stage}. "
                "Expected stage is READY."
            )
        actors = [actor.name for actor in self.graph.vertices]
        res = await invoke_actors_t(remote_call.start, actors)
        res.raise_for_errors()

        res = await invoke_actors_t(remote_call.get_stage, actors)
        # It should be RUNNING, but may be FINISHED/FAILED when workload runs too short.
        assert not any(it == WorkerStage.READY for it in res.results), (
            f"Start should update stage, not READY. {res.as_dict()}"
        )

        logger.info("Job started successfully.")
        self._task = asyncio.create_task(self._main_loop(), name="job_monitor")
        self._update_stage(MasterStage.RUNNING)

    async def _main_loop(self):
        """Monitor the actors' status."""
        while self.stage == MasterStage.RUNNING:
            await asyncio.sleep(5)
            res = await invoke_actors_t(
                remote_call.get_stage, self.graph.vertices
            )
            if all(it in ["FAILED", "FINISHED"] for it in res.results):
                if all(it == "FINISHED" for it in res.results):
                    self.request_stop("All nodes finished successfully.")
                else:
                    self._context.status.exit_code = 1
                    self.request_stop(
                        "All nodes finished, but some nodes failed."
                    )
                break

        assert self.stage == MasterStage.STOPPING, (
            f"Job stage should be STOPPING, but got {self.stage}."
        )
        kill_actors([node.name for node in self.graph.vertices])
        logger.info("Job stopped successfully.")
        self._update_stage(MasterStage.STOPPED)
        self._stopped_event.set()

    async def relaunch_nodes_by_actors(self, actors, wait_interval=10):
        """Relaunch the node if actor exceed restarting limit."""

        # aggregate the target nodes need to be relaunched
        target_relaunch_nodes = list(
            set([actor.node_info for actor in actors])
        )

        # get current nodes from ray(gcs)
        current_nodes = []
        for ray_node in ray.nodes():
            current_nodes.append(
                NodeInfo(
                    id=ray_node["NodeID"],
                    hostname=ray_node["NodeManagerHostname"],
                    ip_address=ray_node["NodeManagerAddress"],
                )
            )
        logger.debug(
            f"Current nodes(size: {len(current_nodes)}): {current_nodes}"
        )

        # relaunch ray node
        logger.info(f"Relaunch nodes: {target_relaunch_nodes}.")
        self.scaler.relaunch(target_relaunch_nodes)

        # wait for nodes relaunching
        await asyncio.wait_for(
            self._wait_node_relaunch(
                target_relaunch_nodes, current_nodes, wait_interval
            ),
            len(target_relaunch_nodes) * RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME,
        )

        # reset actor per node failure count if actor's ray node has relaunched
        for relaunched_node_actor in actors:
            relaunched_node_actor.reset_per_node_failure_count()
            logger.info(
                f"Reset actor {relaunched_node_actor.name} restart "
                "count due to node relaunch."
            )

    def _update_node_restart_count(self, inc_count):
        self._context.node_restart_count += inc_count

    async def _wait_node_relaunch(
        self, relaunch_nodes, current_nodes, wait_interval=10
    ):
        current_nodes_id = [node.id for node in current_nodes]

        def get_relaunched_num():
            new_nodes_id = [ray_node["NodeID"] for ray_node in ray.nodes()]
            return len(set(new_nodes_id) - set(current_nodes_id))

        current_relaunched_num = get_relaunched_num()
        while current_relaunched_num < len(relaunch_nodes):
            logger.info(
                "Waiting for nodes relaunching, "
                f"total relaunching: {len(relaunch_nodes)}, "
                f"finish relaunching: {current_relaunched_num}."
            )
            await asyncio.sleep(wait_interval)
            current_relaunched_num = get_relaunched_num()

        self._update_node_restart_count(len(relaunch_nodes))
        logger.info(f"All target nodes relaunched: {relaunch_nodes}")

    async def restart_actors(self, actors: List[DLExecutionVertex]) -> None:
        """Restart the specified actors."""
        # judge whether to do node relaunching if there is actor restarted due
        # to failure and the restart count exceed limit
        exceed_limit_actors = [
            actor
            for actor in actors
            if actor.per_node_failure_count > actor.spec.per_node_max_failure
        ]
        influenced_actors = []
        if exceed_limit_actors:
            if (
                self._context.node_restart_count
                >= self._context.node_total_count
            ):
                logger.fatal(
                    f"Node relaunch beyond limit: {self._context.node_total_count}, stop job directly."
                )
                self.request_stop("node relaunch beyond limit.")
                return

            # node relaunch may cause more actors restarting, so need to get
            # these influenced actors 1st
            influenced_actors = (
                self.graph.get_all_actors_by_specified_node_actors(
                    exceed_limit_actors
                )
            )

            try:
                await self.relaunch_nodes_by_actors(exceed_limit_actors)
            except asyncio.TimeoutError:
                logger.error("Timeout relaunching ray nodes.")

        # union influenced for set restarting flag
        actors = list(set(actors + influenced_actors))

        for actor in actors:
            actor.inc_restart_count()
            actor.set_restarting()

        # diff_actors = all restart actors - node relaunch actors
        diff_actors = list(set(actors) - set(influenced_actors))

        logger.info(
            f"Positive restarting actors: {[actor.name for actor in diff_actors]}, "
            f"passive restarting actors: {[actor.name for actor in influenced_actors]}"
        )

        # restart diff actors only(because node relaunch will restart
        # other actors automatically)
        await restart_actors([actor.name for actor in diff_actors])
        logger.info({n.name: n.restarting for n in self.graph.vertices})

        await self._wait_ready(actors)
        logger.info({n.name: n.restarting for n in self.graph.vertices})
        for actor in actors:
            actor.set_running()
        logger.info(f"Restarted actors: {[actor.name for actor in actors]}")

    async def deal_with_actor_restarting(self, actor: DLExecutionVertex):
        """
        The core logic of this method should only be executed for actors that
        failed due to exceptions.

        Other actors that are actively or passively restarted for other
        reasons should be filtered out by the following `actor.restarting`
        logic.
        """

        if actor.restarting:
            return  # Actor is already restarting, no need to handle it again.

        # record restarted(failure) actor
        actor.inc_failure_count()

        if (
            actor.spec.is_role_level_failover_supported()
            and actor.role.sub_master is not None
        ):
            # call sub master do role level failover
            await invoke_actor_t(
                remote_call.restart_workers,
                actor.role.sub_master.name,
            )
            return
        else:
            # call job restart
            await self.restart_job()

    async def restart_job(self):
        """Restart the job execution."""
        if self.stage != MasterStage.RUNNING:
            raise RuntimeError(
                f"Cannot restart job in stage {self.stage}. "
                "Expected stage is RUNNING."
            )
        self._context.status.job_restart_count += 1
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
        if self.stage == MasterStage.RUNNING:
            self._update_stage(MasterStage.STOPPING)
        else:
            # No running job, terminate
            kill_actors([node.name for node in self.graph.vertices])
            self._update_stage(MasterStage.STOPPED)
            self._stopped_event.set()

    async def wait(self):
        """Wait for the job to finish."""
        await self._stopped_event.wait()
        assert self.stage == MasterStage.STOPPED

    def _get_state_key(self):
        return MasterConstant.MASTER_STATE_KEY_PREFIX + self.config.job_name

    def save(self):
        """Save the job state to persistent storage."""
        # This is a placeholder for saving the job state.
        # In a real implementation, this would save to a database or file.

        state_key = self._get_state_key()
        self._state_backend.set(state_key, self._context.serialize())

        logger.info("Save runtime context into state.")

    def _init_or_reload(self) -> Tuple[bool, ManagerRuntimeContext]:
        state_key = self._get_state_key()

        if self._state_backend.exists(state_key):
            # reload context
            logger.info(
                "Reload state from state backend to recover runtime context."
            )
            return False, ManagerRuntimeContext.deserialize(
                self._state_backend.get(state_key)
            )
        else:
            # init context
            logger.info("Init runtime context for beginning.")
            return True, ManagerRuntimeContext(
                graph=DLExecutionGraph.create(self.config.dl_config),
                stage=MasterStage.INIT,
                status=MasterStatus(MasterStage.INIT),
                node_total_count=len(ray.nodes()),
            )
