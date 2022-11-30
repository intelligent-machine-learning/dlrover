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

import copy
import itertools
import json
import threading
import base64
from typing import Dict

from kubernetes.client import V1EnvVar

from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.scheduler.kubernetes import k8sClient
from dlrover.python.master.scaler.base_scaler import Scaler
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.common.constants import NodeType
from dlrover.python.common.node import NodeGroupResource

SCALER_GROUP = "elastic.iml.github.io"
SCALER_VERION = "v1alpha1"
SCALER_KIND = "Scaler"


class BashCommandTemplate(object):
    REDIRECTION = "({}) 2>&1 | tee {}"
    SET_PIPEFAIL = "set -o pipefail;"


class PodScaler(Scaler):
    def __init__(self, job_name, namespace, cluster, client: k8sClient):
        super(PodScaler, self).__init__(job_name)
        self._namespace = namespace
        self._cluster = cluster
        self._client = client

    def scale(self, plan: ResourcePlan):
        pass

    def scale_up_pods(self):
        pass

    def scale_down_pods(self):
        pass

    def remove_pods(self):
        pass


class PodAdditionConfig(object):
    def __init__(
        self,
        volume=None,
        image_pull_policy=None,
        restart_policy="Never",
        envs=None,
        log_file_path=None,
    ):
        self.volume = volume
        self.image_pull_policy = image_pull_policy
        self.restart_policy = restart_policy
        self.envs = envs
        self.log_file_path = log_file_path


def new_tf_config(
    group_resource: Dict[str, NodeGroupResource],
    get_service_fn,
    type_key,
    index_key,
    ps_addrs
):
    """Get tf.estimator config spec data. The detail is in
    https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
    """
    cluster_dict = {}
    cluster_dict["ps"] = ps_addrs
    worker_num = group_resource[NodeType.WORKER].count
    if type_key == NodeType.WORKER and index_key >= worker_num:
        worker_num = index_key + 1
    if worker_num > 0:
        cluster_dict["worker"] = []
        for worker_id in range(worker_num):
            cluster_dict["worker"].append(
                get_service_fn(NodeType.WORKER, worker_id)
            )
    evaluator_num = group_resource[NodeType.EVALUATOR].count
    if evaluator_num > 0:
        cluster_dict["evaluator"] = []
        for worker_id in range(evaluator_num):
            cluster_dict["evaluator"].append(
                get_service_fn(NodeType.EVALUATOR, worker_id)
            )
    chief_num = group_resource[NodeType.CHIEF].count
    if chief_num > 0:
        cluster_dict["chief"] = []
        for worker_id in range(chief_num):
            cluster_dict["chief"].append(
                get_service_fn(NodeType.CHIEF, worker_id)
            )

    task_dict = {}
    task_dict["type"] = type_key
    task_dict["index"] = index_key
    return {"cluster": cluster_dict, "task": task_dict}


class PodLauncher(object):
    def __init__(
        self,
        k8s_client,
        command_args,
        need_tf_config,
        pod_addition_config: PodAdditionConfig,
    ):
        self._k8s_client = k8s_client
        self._command_args = command_args
        self._need_tf_config = need_tf_config
        self._volume = pod_addition_config.volume
        self._image_pull_policy = pod_addition_config.image_pull_policy
        self._restart_policy = pod_addition_config.restart_policy
        self._envs = pod_addition_config.envs
        self._log_file_path = pod_addition_config.log_file_path

    def _get_annotation(self, key):
        master_pod = self._k8s_client.get_master_pod()
        if master_pod and master_pod.metadata.annotations:
            cmd_base64 = master_pod.metadata.annotations.get(key, "")
            return base64.b64decode(cmd_base64).decode("utf-8")
        else:
            return None

    def _complement_job_command(self, pod_args, ps_node_argument):
        # pod_args has 2 strings. The first string is "-c" and
        # the second string is the shell command to run, like
        # ["-c", "python -m main --minibatch_size 64"]
        job_command = pod_args[1]
        if ps_node_argument:
            job_command += ps_node_argument
        if self._log_file_path:
            job_command = BashCommandTemplate.REDIRECTION.format(
                job_command, self._log_file_path
            )
        job_command += " ".join(pod_args[2:])
        job_command = BashCommandTemplate.SET_PIPEFAIL + job_command
        return job_command

    
