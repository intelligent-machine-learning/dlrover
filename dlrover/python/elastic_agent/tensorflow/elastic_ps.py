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

import json
import os

from dlrover.python.common.constants import (
    PSClusterVersionType,
    TrainingLoopStatus,
)
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor

monitor = ResourceMonitor()


def get_task_type_from_tf_config():
    if os.getenv("TF_CONFIG", None):
        config = json.loads(os.getenv("TF_CONFIG"))
        if "task" in config and "type" in config["task"]:
            return config["task"]["type"]
    return None


def get_task_id_from_tf_config():
    if os.getenv("TF_CONFIG", None):
        config = json.loads(os.getenv("TF_CONFIG"))
        if "task" in config and "index" in config["task"]:
            return config["task"]["index"]
    return None


class ElasticPsClient(object):
    def __init__(self, task_type=None, task_id=None):
        if not task_type:
            task_type = get_task_type_from_tf_config()
        if not task_id:
            task_id = int(get_task_id_from_tf_config())

        self._task_type = task_type
        self._task_id = task_id
        self._master_client = GlobalMasterClient.MASTER_CLIENT

    def get_global_cluster_version(self):
        response = self._master_client.get_cluster_version(
            PSClusterVersionType.GLOBAL, self._task_type, self._task_id
        )
        return response.version

    def get_local_cluster_version(self):
        response = self._master_client.get_cluster_version(
            PSClusterVersionType.LOCAL, self._task_type, self._task_id
        )
        return response.version

    def update_local_cluster_version(self, version):
        self._master_client.update_cluster_version(
            PSClusterVersionType.LOCAL, version, self._task_type, self._task_id
        )

    def update_global_cluster_version(self, version):
        self._master_client.update_cluster_version(
            PSClusterVersionType.GLOBAL,
            version,
            self._task_type,
            self._task_id,
        )

    def get_restored_version(self):
        response = self._master_client.get_cluster_version(
            PSClusterVersionType.RESTORED, self._task_type, self._task_id
        )
        return response.version

    def update_restored_version(self, version):
        self._master_client.update_cluster_version(
            PSClusterVersionType.RESTORED,
            version,
            self._task_type,
            self._task_id,
        )

    def get_all_ps_nodes(self):
        return self._master_client.query_ps_nodes()

    def ready_for_ps_relaunch(self):
        return self._master_client.ready_for_ps_relaunch()

    def training_started(self):
        status = self._master_client.query_training_status()
        return status == TrainingLoopStatus.START
