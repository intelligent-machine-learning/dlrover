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
from unittest import mock

import yaml
from kubernetes import client

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    NodeStatus,
    NodeType,
    PlatformType,
)
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.shard.dataset_splitter import new_dataset_splitter
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.scheduler.job import JobArgs, NodeArgs
from dlrover.python.scheduler.kubernetes import k8sClient


def get_test_scale_plan(*args, **kwargs):
    with open("dlrover/python/tests/data/scaleplan_sample.yaml", "r") as f:
        job_content = f.read()
    job = yaml.safe_load(job_content)
    return job


def _get_training_job(*args, **kwargs):
    with open("dlrover/python/tests/data/elasticjob_sample.yaml", "r") as f:
        job_content = f.read()
    job = yaml.safe_load(job_content)
    return job


def _get_pod(name):
    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        spec={},
        metadata=client.V1ObjectMeta(
            name=name,
            labels={},
            namespace="default",
            uid="111",
        ),
    )
    return pod


class MockK8sPSJobArgs(JobArgs):
    def __init__(self):
        super(MockK8sPSJobArgs, self).__init__(
            PlatformType.KUBERNETES, "default", "test"
        )

    def initilize(self):
        worker_resource = NodeGroupResource(3, NodeResource(1, 4096))
        self.node_args[NodeType.WORKER] = NodeArgs(
            worker_resource, True, 3, 0, ""
        )

        ps_resource = NodeGroupResource(3, NodeResource(1, 4096))
        self.node_args[NodeType.PS] = NodeArgs(ps_resource, True, 1, 0, "all")

        evaluator_resource = NodeGroupResource(1, NodeResource(1, 4096))
        self.node_args[NodeType.EVALUATOR] = NodeArgs(
            evaluator_resource, False, 1, 0, ""
        )

        chief_resource = NodeGroupResource(1, NodeResource(1, 4096))
        self.node_args[NodeType.CHIEF] = NodeArgs(
            chief_resource, True, 1, 0, ""
        )
        self.job_uuid = "11111"
        self.distribution_strategy = DistributionStrategy.PS


class MockK8sAllreduceJobArgs(JobArgs):
    def __init__(self):
        super(MockK8sAllreduceJobArgs, self).__init__(
            PlatformType.KUBERNETES, "default", "test"
        )

    def initilize(self):
        worker_resource = NodeGroupResource(
            16, NodeResource(1, 4096, "a100", 8)
        )
        self.node_args[NodeType.WORKER] = NodeArgs(
            worker_resource, True, 3, 0, ""
        )
        self.job_uuid = "11111"
        self.distribution_strategy = DistributionStrategy.ALLREDUCE


class MockRayJobArgs(JobArgs):
    def __init__(self):
        super(MockRayJobArgs, self).__init__(
            PlatformType.RAY, "default", "test"
        )

    def initilize(self):
        worker_resource = NodeGroupResource(3, NodeResource(1, 4096))
        self.node_args[NodeType.WORKER] = NodeArgs(
            worker_resource, True, 3, 0, ""
        )

        ps_resource = NodeGroupResource(3, NodeResource(1, 4096))
        self.node_args[NodeType.PS] = NodeArgs(ps_resource, True, 1, 0, "all")

        evaluator_resource = NodeGroupResource(1, NodeResource(1, 4096))
        self.node_args[NodeType.EVALUATOR] = NodeArgs(
            evaluator_resource, False, 1, 0, ""
        )

        chief_resource = NodeGroupResource(1, NodeResource(1, 4096))
        self.node_args[NodeType.CHIEF] = NodeArgs(
            chief_resource, True, 1, 0, ""
        )
        self.job_uuid = "11111"


def create_pod(labels):
    status = client.V1PodStatus(
        container_statuses=[
            client.V1ContainerStatus(
                image="test",
                name="main",
                ready=True,
                restart_count=1,
                image_id="test",
                state=client.V1ContainerState(
                    running=client.V1ContainerStateRunning(
                        started_at=datetime.datetime.strptime(
                            "2022-11-11 11:11:11", "%Y-%m-%d %H:%M:%S"
                        ),
                    )
                ),
            )
        ],
        phase=NodeStatus.RUNNING,
    )

    resource = {"cpu": 1, "memory": "10Gi"}
    container = client.V1Container(
        name="main",
        image="test",
        command="echo 1",
        resources=client.V1ResourceRequirements(
            requests=resource,
            limits=resource,
        ),
        image_pull_policy="Never",
    )

    # Pod
    spec = client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        priority_class_name="high",
    )

    pod = client.V1Pod(
        api_version="v1",
        kind="Pod",
        spec=spec,
        metadata=client.V1ObjectMeta(
            name="test-worker-0",
            labels=labels,
        ),
        status=status,
    )
    return pod


def mock_list_namespaced_pod(label_selector):
    pods = []
    for i in range(2):
        labels = {
            ElasticJobLabel.APP_NAME: "test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.PS,
            ElasticJobLabel.REPLICA_INDEX_KEY: str(i),
            ElasticJobLabel.RANK_INDEX_KEY: str(i),
        }
        pod = create_pod(labels)
        pods.append(pod)

    for i in range(3):
        labels = {
            ElasticJobLabel.APP_NAME: "test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.WORKER,
            ElasticJobLabel.REPLICA_INDEX_KEY: str(i),
            ElasticJobLabel.RANK_INDEX_KEY: str(i),
        }
        pod = create_pod(labels)
        pods.append(pod)
    return client.V1PodList(
        items=pods, metadata=client.V1ListMeta(resource_version="12345678")
    )


def create_task_manager(dataset_name="test"):
    task_manager = TaskManager(False, SpeedMonitor())
    splitter = new_dataset_splitter(
        False,
        100,
        1000,
        1,
        dataset_name,
        "table",
    )
    task_manager.new_dataset(
        batch_size=10,
        dataset_size=1000,
        dataset_name=dataset_name,
        dataset_splitter=splitter,
        task_type=elastic_training_pb2.TRAINING,
    )
    return task_manager


def mock_k8s_client():
    k8s_client = k8sClient.singleton_instance("default")
    k8s_client.get_custom_resource = _get_training_job  # type: ignore
    k8s_client.get_pod = _get_pod  # type: ignore
    k8s_client.list_namespaced_pod = mock_list_namespaced_pod  # type: ignore
    k8s_client.create_custom_resource = mock.MagicMock(  # type: ignore
        return_value=True
    )
    k8s_client.delete_custom_resource = mock.MagicMock(  # type: ignore
        return_value=True
    )
    k8s_client.create_pod = mock.MagicMock(return_value=True)  # type: ignore
    k8s_client.create_service = mock.MagicMock(  # type: ignore
        return_value=True
    )
