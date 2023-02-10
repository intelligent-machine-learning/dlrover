# Copyright 2022 The EasyDL Authors. All rights reserved.
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

import datetime
import unittest
from typing import List

from kubernetes import client

from dlrover.python.common.constants import (
    ElasticJobLabel,
    NodeExitReason,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.node import Node
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.watcher.base_watcher import NodeEvent
from dlrover.python.master.watcher.k8s_watcher import (
    K8sScalePlanWatcher,
    PodWatcher,
    _convert_pod_event_to_node_event,
    _get_pod_exit_reason,
)
from dlrover.python.tests.test_utils import (
    create_pod,
    get_test_scale_plan,
    mock_k8s_client,
)


class PodWatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_list(self):
        mock_k8s_client()
        pod_watcher = PodWatcher("test", "")
        nodes: List[Node] = pod_watcher.list()
        self.assertEqual(len(nodes), 5)
        node: Node = nodes[0]
        self.assertEqual(node.id, 0)
        self.assertEqual(node.type, NodeType.PS)
        self.assertEqual(node.status, NodeStatus.RUNNING)
        self.assertEqual(
            node.start_time,
            datetime.datetime.strptime(
                "2022-11-11 11:11:11", "%Y-%m-%d %H:%M:%S"
            ),
        )
        node: Node = nodes[-1]
        self.assertEqual(node.id, 2)
        self.assertEqual(node.type, NodeType.WORKER)
        self.assertEqual(node.status, NodeStatus.RUNNING)

    def test_convert_pod_event_to_node_event(self):
        labels = {
            ElasticJobLabel.APP_NAME: "test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.WORKER,
            ElasticJobLabel.REPLICA_INDEX_KEY: "0",
            ElasticJobLabel.RANK_INDEX_KEY: "0",
        }
        pod = create_pod(labels)
        event_type = "Modified"
        event = {"object": pod, "type": event_type}
        node_event: NodeEvent = _convert_pod_event_to_node_event(event)
        self.assertEqual(node_event.event_type, event_type)
        self.assertEqual(node_event.node.id, 0)
        self.assertEqual(node_event.node.type, NodeType.WORKER)
        self.assertEqual(node_event.node.config_resource.cpu, 1)
        self.assertEqual(node_event.node.config_resource.memory, 10240)

    def test_get_pod_exit_reason(self):
        labels = {
            ElasticJobLabel.APP_NAME: "test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.WORKER,
            ElasticJobLabel.REPLICA_INDEX_KEY: "0",
            ElasticJobLabel.RANK_INDEX_KEY: "0",
        }
        pod = create_pod(labels)
        state = pod.status.container_statuses[0].state
        state.terminated = client.V1ContainerStateTerminated(
            reason="OOMKilled",
            exit_code=143,
        )
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.OOM)
        state.terminated = client.V1ContainerStateTerminated(exit_code=137)
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.KILLED)
        state.terminated = client.V1ContainerStateTerminated(exit_code=1)
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.FATAL_ERROR)


class ScalePlanWatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_get_resource_plan_from_scale_plan(self):
        watcher = K8sScalePlanWatcher("test", "default", "1234")
        scale_plan = get_test_scale_plan()
        resource_plan: ResourcePlan = watcher._get_resoruce_plan_from_event(
            scale_plan
        )
        self.assertEqual(
            resource_plan.node_group_resources[NodeType.WORKER].count, 2
        )
        self.assertEqual(
            resource_plan.node_group_resources[
                NodeType.WORKER
            ].node_resource.cpu,
            0.5,
        )
        self.assertEqual(
            resource_plan.node_group_resources[
                NodeType.WORKER
            ].node_resource.memory,
            256,
        )
        self.assertEqual(
            resource_plan.node_group_resources[NodeType.PS].count, 1
        )
        self.assertEqual(len(resource_plan.node_resources), 2)
        self.assertEqual(
            resource_plan.node_resources["elasticjob_sample-ps-0"].cpu, 4.0
        )
        self.assertEqual(
            resource_plan.node_resources["elasticjob_sample-ps-0"].memory, 1024
        )
        self.assertEqual(
            resource_plan.node_resources["elasticjob_sample-worker-0"].cpu, 4.0
        )
        self.assertEqual(
            resource_plan.node_resources["elasticjob_sample-worker-0"].memory,
            1024,
        )
