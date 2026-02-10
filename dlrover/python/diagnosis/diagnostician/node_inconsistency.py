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
from typing import Optional, Dict

from dlrover.python.common.constants import (
    EventReportConstants,
    ElasticJobLabel,
    NodeStatus,
)
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisErrorConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.scheduler.kubernetes import k8sClient
from dlrover.python.util import k8s_util


class NodeInconsistencyDiagnostician(Diagnostician):
    """
    NodeInconsistencyDiagnostician detect following conditions:
    1. Same node but multiple instances.
    2. TODO
    """

    def __init__(self, job_args):
        super().__init__()
        self._job_args = job_args
        self._k8s_client = k8sClient.singleton_instance(job_args.namespace)

    # by type+index
    def _get_pod_unique_labels(self, node: Node):
        return {
            ElasticJobLabel.JOB_KEY: self._job_args.job_name,
            ElasticJobLabel.REPLICA_TYPE_KEY: node.type,
            ElasticJobLabel.RANK_INDEX_KEY: node.rank_index,
        }

    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        job_nodes: Dict[str, Dict[int, Node]] = kwargs["job_nodes"]
        running_nodes = []
        for _, nodes in job_nodes.items():
            for _, node in nodes.items():
                if not node.is_released and node.status == NodeStatus.RUNNING:
                    running_nodes.append(node)

        for running_node in running_nodes:
            pod_labels_selector = k8s_util.gen_k8s_label_selector_from_dict(
                self._get_pod_unique_labels(running_node)
            )
            pods = self._k8s_client.list_namespaced_pod(pod_labels_selector)
            if (
                pods
                and len(pods.items) > 1
                and all(
                    pod.status.phase == NodeStatus.RUNNING
                    and not pod.metadata.deletion_timestamp
                    for pod in pods.items
                )
            ):
                return DiagnosisObservation(
                    DiagnosisErrorConstant.REPEATED_NODE,
                    {"target": pod_labels_selector},
                )

            # TODO: more inconsistency case if needed

        logger.debug("No inconsistency node found.")
        return None

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        if problem.observation in [DiagnosisErrorConstant.REPEATED_NODE]:
            return EventAction(
                event_type=EventReportConstants.TYPE_WARN,
                event_instance=f"{DiagnosisConstant.MASTER_INSTANCE}",
                event_action=problem.observation,
                event_msg=problem.extra_infos.get("target", ""),
                event_labels={},
                expired_time_period=120,
            )
        else:
            return NoAction()
