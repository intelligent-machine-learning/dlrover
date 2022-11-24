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

from kubernetes import client

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    NodeStatus,
    NodeType,
)
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.shard_manager.task_manager import TaskManager


class MockArgs(object):
    def __init__(self):
        self.job_name = "test"
        self.namespace = "test"
        self.ps_is_critical = True
        self.ps_relaunch_max_num = 1
        self.use_ddp = False
        self.critical_worker_index = "0:3"
        self.distribution_strategy = DistributionStrategy.PARAMETER_SERVER
        self.relaunch_on_worker_failure = 1
        self.num_workers = 3
        self.worker_resource_request = "cpu=1,memory=4096Mi"
        self.worker_pod_priority = ""

        self.num_ps_pods = 3
        self.ps_resource_request = "cpu=1,memory=4096Mi"
        self.ps_pod_priority = ""

        self.num_evaluators = 1
        self.evaluator_resource_request = "cpu=1,memory=4096Mi"
        self.evaluator_pod_priority = ""

        self.num_tf_master = 3
        self.tf_master_resource_request = "cpu=1,memory=4096Mi"
        self.tf_master_pod_priority = ""

        self.need_node_manager = True
        self.need_task_manager = True
        self.relaunch_timeout_worker = False
        self.cluster = "local"
        self.user = "dlrover"
        self.port = 2222


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
    pod = client.V1Pod(
        kind="Pod",
        metadata=client.V1ObjectMeta(
            labels=labels,
        ),
        status=status,
    )
    return pod


def mock_list_job_pods():
    pods = []
    for i in range(2):
        labels = {
            ElasticJobLabel.APP_NAME: "test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.PS,
            ElasticJobLabel.REPLICA_INDEX_KEY: str(i),
            ElasticJobLabel.TRAINING_TASK_INDEX_KEY: str(i),
        }
        pod = create_pod(labels)
        pods.append(pod)

    for i in range(3):
        labels = {
            ElasticJobLabel.APP_NAME: "test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.WORKER,
            ElasticJobLabel.REPLICA_INDEX_KEY: str(i),
            ElasticJobLabel.TRAINING_TASK_INDEX_KEY: str(i),
        }
        pod = create_pod(labels)
        pods.append(pod)
    return client.V1PodList(
        items=pods, metadata=client.V1ListMeta(resource_version="12345678")
    )


def create_task_manager():
    task_manager = TaskManager(False, SpeedMonitor())
    dataset_name = "test"
    task_manager.new_dataset(
        batch_size=10,
        num_epochs=1,
        dataset_size=1000,
        shuffle=False,
        num_minibatches_per_shard=10,
        dataset_name=dataset_name,
        task_type=elastic_training_pb2.TRAINING,
        storage_type="table",
    )
    return task_manager
