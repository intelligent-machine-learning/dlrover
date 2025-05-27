#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import threading
import time
from datetime import datetime
from typing import Dict, List

import ray

from dlrover.python.common.constants import PlatformType, NodeType, NodeStatus, \
    NodeEventType, JobStage
from dlrover.python.common.global_context import DefaultValues
from dlrover.python.common.node import NodeEvent, Node, NodeResource
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction, \
    NoAction
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.dist_job_manager import DistributedJobManager
from dlrover.python.master.node.event_callback import NodeEventCallback
from dlrover.python.master.node.training_node import get_critical_worker_index, \
    set_critical_node, update_nodes_priority
from dlrover.python.master.watcher.factory import new_node_watcher
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.enums import InternalRoleType
from dlrover.python.unified.master.elastic.job_context import \
    get_elastic_job_context
from dlrover.python.unified.master.job_manager import JobManager

_MAX_POD_RELAUNCH_COUNT = 5


class ElasticJobManager(JobManager):
    """
    ElasticJobManager is the ray based extension for handling
    torch elastic training.
    """

    def __init__(
        self,
        job_args: JobArgs,
        perf_monitor: PerfMonitor,
        external_config=None,
    ):
        self._elastic_job_context = get_elastic_job_context()
        self._lock = threading.Lock()

        # self._remove_exited_node = job_args.remove_exited_node
        # node_restart_count: Dict[str, int] = {}
        # for node_type, node_args in job_args.node_args.items():
        #     self._job_resource.node_group_resources[
        #         type
        #     ] = node_args.group_resource
        #     node_restart_count[node_type] = node_args.restart_count
        #
        # self._relaunch_on_worker_failure = min(
        #     node_restart_count.get(NodeType.WORKER, 0), _MAX_POD_RELAUNCH_COUNT
        # )
        # self._critical_worker_index = get_critical_worker_index(job_args)
        # logger.info(
        #     f"Worker relaunch number: {self._relaunch_on_worker_failure}; "
        #     f"Critical worker index: {self._critical_worker_index}."
        # )
        #
        # self._node_event_callbacks: List[NodeEventCallback] = []
        # self._node_watcher = new_node_watcher(
        #     PlatformType.RAY, job_args.job_name, job_args.namespace
        # )
        # self._init_training_node_manager()


    @property
    def elastic_context(self):
        return self._elastic_job_context

    def start(self):
        # no need to start thread to monitor
        pass

    def _init_nodes(self):
        # init nodes from computation graph
        job_nodes: Dict[str, Dict[int, Node]] = {}
        group_nodes: Dict[int, Node] = {}

        for elastic_vertic in self._elastic_job_context.graph().execution_vertices[InternalRoleType.ELASTIC.name]:
            group_nodes[elastic_vertic.rank] = Node(
                node_type=NodeType.WORKER,
                node_id=elastic_vertic.rank,
                rank_index=elastic_vertic.rank,
                name=elastic_vertic.name,
                config_resource=NodeResource.resource_to_node_resource(elastic_vertic.resource),
                max_relaunch_count=elastic_vertic.get_extra_args("max_relaunch_count", 3),
                service_addr=elastic_vertic.actor_handle,
            )
        job_nodes[NodeType.WORKER] = group_nodes
        set_critical_node(
            job_nodes,
            critical_worker_index=self._critical_worker_index,
        )
        update_nodes_priority(job_nodes)
        self._elastic_job_context.update_job_nodes(job_nodes)

    def update_node_required_info(self, min_required, max_required, timeout):
        pass

    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        pass

    def update_node_paral_config(self, node_type, node_id, paral_config):
        node = self.elastic_context.job_node(node_type, node_id)
        if node is None:
            logger.warning(f"not found Node[{node_type}][{node_id}]")
            return
        node.update_paral_config(paral_config)
        self.elastic_context.update_job_node(node)

    def collect_node_heart_beat(
        self, node_type, node_id, timestamp
    ) -> DiagnosisAction:
        with self._lock:
            node = self.elastic_context.job_node(node_type, node_id)
            if node is None:
                return NoAction()
            if node.heartbeat_time == 0:
                logger.info(f"Start receiving heartbeat from node {node_id}")
            node.heartbeat_time = timestamp
            self.elastic_context.update_job_node(node)
            action = self.elastic_context.next_action(instance=node_id)
            if not action or isinstance(action, NoAction):
                return self.elastic_context.next_action(
                    instance=DiagnosisConstant.ANY_INSTANCE
                )
            else:
                logger.debug(f"Collect action from {node_id}: {action}")
                return action

    def get_running_nodes(self):
        nodes = []
        with self._lock:
            worker_nodes = self.elastic_context.job_nodes_by_type(NodeType.WORKER)
            for node in worker_nodes.values():
                if node.status == NodeStatus.RUNNING:
                    nodes.append(node)
        return nodes

    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        node = self.elastic_context.job_node(node_type, node_id)
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
                    now = datetime.now()
                    node.start_hang_time = now.timestamp()
            else:
                node.start_hang_time = 0
            self.elastic_context.update_job_node(node)
        else:
            logger.warning(
                "CPU requests not configure "
                "and can not determine if the job node is hung"
            )

    def process_reported_node_event(self, node_event: NodeEvent):
        """
        The node events here is reported from training agent.

        Args:
            node_event: The event from training agent.
        """

        event_type = node_event.event_type
        node = node_event.node
        node_type = node.type
        node_id = node.id

        with self._lock:
            target_node = self.elastic_context.job_node(node_type, node_id)
            if target_node:
                logger.info(
                    f"Node {node_id}({node_type}) reported "
                    f"status to {event_type}."
                )
                target_node.update_reported_status(event_type)
                self.elastic_context.update_job_node(target_node)

            if event_type == NodeEventType.SUCCEEDED_EXITED:
                self.elastic_context.update_job_stage(JobStage.JOB_STOPPING)
                logger.info(
                    f"Update job stage to {self.elastic_context.get_job_stage()} "
                    f"due to event {event_type}."
                )
