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
import threading
import time
from datetime import datetime
from typing import Dict, List

from dlrover.python.common.constants import (
    JobStage,
    NodeEventType,
    NodeStatus,
    NodeType,
    PlatformType,
    TrainingExceptionLevel,
)
from dlrover.python.common.global_context import DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeEvent, NodeResource
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)

# isort: off
from dlrover.python.master.node.job_context import (
    get_job_context as get_elastic_context,
)

# isort: on
from dlrover.python.master.watcher.factory import new_node_watcher
from dlrover.python.unified.common.enums import InternalRoleType
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.elastic.executor import ElasticExecutor
from dlrover.python.unified.master.elastic.failover import FAILURE_TYPE_KEY
from dlrover.python.unified.master.job_manager import JobManager

_MAX_POD_RELAUNCH_COUNT = 5


class ElasticJobManager(JobManager):
    """
    ElasticJobManager is the ray based extension for handling
    torch elastic training.
    """

    def __init__(self):
        super(ElasticJobManager, self).__init__()

        self._elastic_context = get_elastic_context()
        self._node_watcher = new_node_watcher(PlatformType.RAY, self.job_name)
        self._lock = threading.Lock()

    @property
    def elastic_context(self):
        return self._elastic_context

    def get_executor(self):
        return ElasticExecutor(self.graph)

    def start_job(self):
        self._init_nodes()

        super(ElasticJobManager, self).start_job()

        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()

    def _init_nodes(self):
        # init nodes from computation graph
        job_nodes: Dict[str, Dict[int, Node]] = {}
        group_nodes: Dict[int, Node] = {}

        for elastic_vertex in self.context.execution_graph.execution_vertices[
            InternalRoleType.ELASTIC.name
        ]:
            group_nodes[elastic_vertex.rank] = Node(
                node_type=NodeType.WORKER,
                node_id=elastic_vertex.rank,
                rank_index=elastic_vertex.rank,
                name=elastic_vertex.name,
                config_resource=NodeResource.resource_to_node_resource(
                    elastic_vertex.resource
                ),
                max_relaunch_count=elastic_vertex.get_extra_args(
                    "max_relaunch_count", 3
                ),
                service_addr=elastic_vertex.actor_handle,
            )
        job_nodes[NodeType.WORKER] = group_nodes
        self.elastic_context.update_job_nodes(job_nodes)

    def _monitor_nodes(self):
        logger.info("Start monitoring nodes status...")
        while True:
            if self._stopped:
                logger.info("Stop monitoring nodes.")
                break
            try:
                # update directly
                list_nodes = self._node_watcher.list()
                logger.debug(
                    f"Got list nodes: {list_nodes}, current job "
                    f"nodes: {self.elastic_context.job_nodes()}"
                )
                for list_node in list_nodes:
                    node_id = int(list_node.id)
                    with self._lock:
                        current_node = self.elastic_context.job_node(
                            NodeType.WORKER, node_id
                        )
                        if current_node:
                            current_node.status = list_node.status
                            self.elastic_context.update_job_node(current_node)
                        else:
                            logger.warning(
                                f"Node {node_id} not exists "
                                "during monitoring."
                            )
            except Exception as e:
                logger.warning(e)
                time.sleep(30)
            time.sleep(5)

    def update_node_required_info(self, min_required, max_required, timeout):
        # TODO
        pass

    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        # TODO
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
            worker_nodes = self.elastic_context.job_nodes_by_type(
                NodeType.WORKER
            )
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
                    "Update job stage to "
                    f"{self.elastic_context.get_job_stage()} "
                    f"due to event {event_type}."
                )

    def has_job_error(self):
        return len(self.executor.get_error()) > 0

    def gen_failures_by_error(self) -> List[FailureDesc]:
        failures = []
        if self.has_job_error():
            for error_vertex in self.executor.get_error():
                failures.append(
                    FailureDesc(
                        workload_name=error_vertex,
                        workload_role=InternalRoleType.ELASTIC.name,
                        failure_time=int(time.time()),
                        failure_level=3,
                        extra_info={
                            FAILURE_TYPE_KEY: TrainingExceptionLevel.NODE_ERROR
                        },
                    )
                )
        logger.info(
            f"Generated failures: {failures} by elastic training error."
        )
        return failures

    def re_execute(self, vertex_name):
        self.executor.add_execution(vertex_name)
