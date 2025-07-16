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
import threading
import time
from typing import List

from dlrover.python.common.constants import (
    JobStage,
    NodeEventType,
    NodeStatus,
    NodeType,
    PreCheckStatus,
)
from dlrover.python.common.global_context import DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeEvent
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.unified.common.workload_base import ActorInfo, WorkerStage
from dlrover.python.unified.util.actor_helper import (
    BatchInvokeResult,
    invoke_actors_async,
)

# isort: off
from dlrover.python.master.node.job_context import (
    get_job_context as get_elastic_context,
)

# isort: on


def convert_to_node_state(state: WorkerStage):
    if state in ["INIT", "CHECKING"]:
        return NodeStatus.PENDING
    elif state in ["RUNNING"]:
        return NodeStatus.RUNNING
    elif state in ["FINISHED", "FAILED"]:
        return NodeStatus.FINISHED
    return NodeStatus.UNKNOWN


class ElasticManager:
    def __init__(self, nodes: List[ActorInfo]):
        self.nodes: List[ActorInfo] = nodes
        self.finished = False

        self.perf_monitor = PerfMonitor()
        self.diagnosis = DiagnosisMaster()
        self.rdzv_manager = ElasticTrainingRendezvousManager()
        self.node_check_manager = NetworkCheckRendezvousManager()
        # self.node_watcher = ActorWatcher(self.job_name, "default")

        # This is old singleton context, used for compatibility
        self._lock = threading.Lock()
        self._old_context = get_elastic_context()

    def _prepare(self):
        self._init_old_context()

    def _init_old_context(self):
        group_nodes = {}
        for elastic_vertex in self.nodes:
            group_nodes[elastic_vertex.rank] = Node(
                node_type=NodeType.WORKER,
                node_id=elastic_vertex.rank,
                rank_index=elastic_vertex.rank,
                name=elastic_vertex.name,
                max_relaunch_count=3,
            )
        self._old_context.update_job_nodes({NodeType.WORKER: group_nodes})

    async def check_workers(self):
        logger.info("Do node-check for all nodes...")
        res = await invoke_actors_async(
            [node.name for node in self.nodes], "run_node_check"
        )
        res.raise_for_errors()
        logger.info("Node-check finished for all nodes.")

    async def start(self):
        # Initialize the elastic client here
        logger.info("Start job execution.")
        self.setup_workloads()
        res = await invoke_actors_async(
            [node.name for node in self.nodes], "start_elastic_job"
        )
        res.raise_for_errors()
        res = await invoke_actors_async(
            [node.name for node in self.nodes], "status"
        )
        if any(it != "RUNNING" for it in res.results):
            raise Exception("Some nodes failed to start the job.")
        self._old_context.set_pre_check_status(PreCheckStatus.PASS)
        asyncio.create_task(self._monitor_nodes(), name="monitor_nodes")

    def setup_workloads(self):
        logger.info("Start setup all workloads...")
        start = time.time() * 1000
        # TODO setup rendezvous manager
        ready = []
        end = time.time() * 1000 - start
        logger.info(
            f"Finish setup all workloads({len(ready)}), cost: {end:.2f}ms"
        )

    async def _monitor_nodes(self):
        logger.info("Start monitoring nodes status...")
        while not self.finished:
            try:
                res: BatchInvokeResult[
                    WorkerStage
                ] = await invoke_actors_async(
                    [node.name for node in self.nodes], "status"
                )
                logger.debug(f"Node status results: {res.results}")
                with self._lock:
                    for node, status in zip(self.nodes, res.results):
                        old_node = self._old_context.job_node(
                            NodeType.WORKER, node.rank
                        )
                        assert old_node is not None, (
                            f"Node({node.rank}) not found in context"
                        )
                        status = convert_to_node_state(status)
                        old_node.update_status(status)
                        self._old_context.update_job_node(old_node)
                if all(it.is_terminal() for it in res.results):
                    self.finished = True
                    logger.info("All nodes are finished.")
                    break
            except Exception as e:
                logger.warning(e)
                await asyncio.sleep(30)
            await asyncio.sleep(5)

    def process_reported_node_event(self, node_event: NodeEvent):
        event_type = node_event.event_type
        node = node_event.node
        node_type = node.type
        node_id = node.id
        with self._lock:
            target_node = self._old_context.job_node(node_type, node_id)
            if target_node:
                logger.info(
                    f"Node {node_id}({node_type}) reported "
                    f"status to {event_type}."
                )
                target_node.update_reported_status(event_type)
                self._old_context.update_job_node(target_node)
            if event_type == NodeEventType.SUCCEEDED_EXITED:
                self._old_context.update_job_stage(JobStage.JOB_STOPPING)
                logger.info(
                    "Update job stage to "
                    f"{self._old_context.get_job_stage()} "
                    f"due to event {event_type}."
                )

    def is_job_finished(self):
        return self.finished

    def has_job_error(self):
        return False

    # ====== 节点管理与监控 ======

    def update_node_required_info(self, min_required, max_required, timeout):
        pass

    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        pass

    # ====== 节点状态与资源 ======
    def update_node_paral_config(self, node_type, node_id, paral_config):
        node = self._old_context.job_node(node_type, node_id)
        if node is None:
            logger.warning(f"not found Node[{node_type}][{node_id}]")
            return
        node.update_paral_config(paral_config)
        self._old_context.update_job_node(node)

    def collect_node_heart_beat(
        self, node_type, node_id, timestamp
    ) -> DiagnosisAction:
        with self._lock:
            node = self._old_context.job_node(node_type, node_id)
            if node is None:
                return NoAction()
            if node.heartbeat_time == 0:
                logger.info(f"Start receiving heartbeat from node {node_id}")
            node.heartbeat_time = timestamp
            self._old_context.update_job_node(node)
            action = self._old_context.next_action(instance=node_id)
            if not action or isinstance(action, NoAction):
                return self._old_context.next_action(
                    instance=DiagnosisConstant.ANY_INSTANCE
                )
            else:
                logger.debug(f"Collect action from {node_id}: {action}")
                return action

    def get_running_nodes(self):
        nodes = []
        with self._lock:
            worker_nodes = self._old_context.job_nodes_by_type(NodeType.WORKER)
            for node in worker_nodes.values():
                if node.status == NodeStatus.RUNNING:
                    nodes.append(node)
        return nodes

    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        node = self._old_context.job_node(node_type, node_id)
        if node is None:
            logger.warning(
                f"Skip update node[{node_type}][{node_id}] resources"
            )
            return
        node.update_resource_usage(cpu, memory, gpu_stats)
        if node.config_resource.cpu:
            cpu_percent = node.used_resource.cpu / node.config_resource.cpu
            if cpu_percent < DefaultValues.HANG_CPU_USAGE_RATE:
                if node.start_hang_time == 0:
                    from datetime import datetime

                    now = datetime.now()
                    node.start_hang_time = now.timestamp()
            else:
                node.start_hang_time = 0
            self._old_context.update_job_node(node)
        else:
            logger.warning(
                "CPU requests not configure "
                "and can not determine if the job node is hung"
            )
