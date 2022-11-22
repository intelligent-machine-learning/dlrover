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
    ElasticJobLabel,
    NodeStatus,
    NodeType,
)
from dlrover.python.master.shard_manager.task_manager import TaskManager


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
    task_manager = TaskManager(False)
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
