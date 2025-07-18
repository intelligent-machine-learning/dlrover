# Copyright 2022 The DLRover Authors. All rights reserved.
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
import json
import os
import time
import unittest
from typing import List
from unittest import mock
from unittest.mock import patch

from kubernetes import client

from dlrover.python.common.constants import (
    ElasticJobLabel,
    JobStage,
    NodeEventType,
    NodeExitReason,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.node import Node, NodeEvent
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.watcher.k8s_watcher import (
    K8sElasticJobWatcher,
    K8sScalePlanWatcher,
    PodWatcher,
    _convert_pod_event_to_node_event,
    _get_pod_exit_reason,
    _verify_restarting_training,
)
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.tests.test_utils import (
    WITH_TO_DELETED,
    create_pod,
    get_test_scale_plan,
    mock_k8s_client,
    mock_list_namespaced_pod,
)


def _mock_pod_labels():
    labels = {
        ElasticJobLabel.APP_NAME: "test",
        ElasticJobLabel.JOB_KEY: "test",
        ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.WORKER,
        ElasticJobLabel.REPLICA_INDEX_KEY: "0",
        ElasticJobLabel.RANK_INDEX_KEY: "0",
        ElasticJobLabel.RELAUNCH_COUNT: "1",
    }
    return labels


class PodWatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.k8s_client = mock_k8s_client()

    def test_list(self):
        # set env
        os.environ[WITH_TO_DELETED] = "True"

        mock_k8s_client()
        pod_watcher = PodWatcher("test", "")
        nodes: List[Node] = pod_watcher.list()
        self.assertEqual(len(nodes), 6)
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
        self.assertIsNotNone(node.host_name)
        self.assertIsNotNone(node.host_ip)
        node: Node = nodes[-2]
        self.assertEqual(node.id, 2)
        self.assertEqual(node.type, NodeType.WORKER)
        self.assertEqual(node.status, NodeStatus.RUNNING)

        node: Node = nodes[-1]
        self.assertEqual(node.id, 99)
        self.assertEqual(node.type, NodeType.WORKER)
        self.assertEqual(node.status, NodeStatus.DELETED)

        # reset env
        os.environ.pop(WITH_TO_DELETED)

    def test_list_succeeded(self):
        pod_watcher = PodWatcher("test", "")

        pods = mock_list_namespaced_pod("")
        pod_watcher._k8s_client.list_namespaced_pod = mock.MagicMock(
            return_value=pods
        )
        pod_watcher._k8s_client.delete_pod = mock.MagicMock(
            side_effect=Exception("test123")
        )
        try:
            pod_watcher.list()
        except Exception:
            self.fail()

        pods.items[0].status.phase = NodeStatus.SUCCEEDED
        try:
            pod_watcher.list()
            self.fail()
        except Exception as e:
            self.assertEqual(e.args[0], "test123")

    def test_convert_pod_event_to_node_event(self):
        labels = _mock_pod_labels()
        pod = create_pod(labels)
        event_type = NodeEventType.MODIFIED
        event = {"object": pod, "type": event_type}
        node_event: NodeEvent = _convert_pod_event_to_node_event(event)
        self.assertEqual(node_event.event_type, event_type)
        self.assertEqual(node_event.node.id, 0)
        self.assertEqual(node_event.node.type, NodeType.WORKER)
        self.assertEqual(node_event.node.config_resource.cpu, 1)
        self.assertEqual(node_event.node.config_resource.memory, 10240)

    def test_get_pod_exit_reason(self):
        labels = _mock_pod_labels()
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

        state.terminated = client.V1ContainerStateTerminated(exit_code=201)
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.HARDWARE_ERROR)

        state.terminated = client.V1ContainerStateTerminated(exit_code=202)
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.HARDWARE_ERROR)

        state.terminated = client.V1ContainerStateTerminated(exit_code=0)
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.Succeeded)

        state.terminated = client.V1ContainerStateTerminated(exit_code=999)
        exit_reason = _get_pod_exit_reason(pod)
        self.assertEqual(exit_reason, NodeExitReason.UNKNOWN_ERROR)

    def test_verify_restarting_training(self):
        labels = _mock_pod_labels()
        pod = create_pod(labels)
        reset = _verify_restarting_training(pod)
        self.assertFalse(reset)
        action = {
            "observedTime": "2020-04-30 00:00:00",
            "scheduledExecutionTime": "2020-04-30 00:10:00",
            "scheduledAction": "RestartTrain_Observe",
            "device_ids": ["npu_id_1", "npu_id_2"],
            "eventType": "NPU_reset",
        }
        pod.metadata.annotations["pod.sigma.ali/scheduled-action"] = (
            json.dumps(action)
        )
        reset = _verify_restarting_training(pod)
        self.assertTrue(reset)


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


class K8sElasticJobWatcherTest(unittest.TestCase):
    def setUp(self):
        mock_k8s_client()
        self.watcher = K8sElasticJobWatcher(JobArgs("k8s", "default", "test"))

    def test_watch_modified_event_suspend(self):
        # 模拟事件流
        event_stream = [
            {
                "type": "MODIFIED",
                "object": {
                    "metadata": {"name": "test"},
                    "spec": {"suspend": True},
                },
            }
        ]

        self.watcher._job_context.update_job_stage(JobStage.JOB_INIT)
        with patch("kubernetes.watch.Watch") as mock_watch:
            mock_watch.return_value.stream.return_value = iter(event_stream)
            self.watcher.start()

        time.sleep(10)

        self.assertIn(
            self.watcher._job_context.get_job_stage(),
            "suspended, stopped, stopping",
        )

    def test_watch_added_event_unsuspend(self):
        event_stream = [
            {
                "type": "ADDED",
                "object": {
                    "metadata": {"name": "test"},
                    "spec": {"suspend": False},
                },
            }
        ]

        self.watcher._job_context.update_job_stage(JobStage.JOB_INIT)
        with patch("kubernetes.watch.Watch") as mock_watch:
            mock_watch.return_value.stream.return_value = iter(event_stream)
            self.watcher.start()

        time.sleep(10)

        self.assertEqual(
            self.watcher._job_context.is_suspended(),
            False,
        )
