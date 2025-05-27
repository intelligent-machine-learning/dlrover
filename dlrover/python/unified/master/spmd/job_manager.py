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
from typing import Dict, List

import ray

from dlrover.python.common.constants import PlatformType, NodeType, NodeStatus, \
    NodeEventType
from dlrover.python.common.node import NodeEvent, Node, NodeResource
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction
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
        # self._lock = threading.Lock()

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
