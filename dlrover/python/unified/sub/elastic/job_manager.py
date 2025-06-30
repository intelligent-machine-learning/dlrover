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

from dlrover.python.common.constants import NodeEventType, NodeStatus, NodeType
from dlrover.python.common.global_context import DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeEvent
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.hybrid.elastic.manager import ElasticManager
from dlrover.python.unified.common.enums import JobStage


# TODO: merge into ElasticManager
class ElasticJobManager(ElasticManager):
    def is_job_finished(self):
        return self.executor.is_finished() and self.stage != JobStage.FAILOVER

    def has_job_error(self):
        return len(self.executor.get_error()) > 0

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

    def process_reported_node_event(self, node_event: NodeEvent):
        event_type = node_event.event_type
        node = node_event.node
        node_type = node.id
        node_id = node.id
        with self._lock:
            target_node = self._old_context.job_node(node_type, node_id)
            if target_node:
                logger.info(
                    f"Node {node_id}({node_type}) reported status to {event_type}."
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
