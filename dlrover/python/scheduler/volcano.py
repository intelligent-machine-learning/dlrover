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

import os
import time

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeEnv,
    NodeType,
    OptimizeMode,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.scheduler.job import ElasticJob, JobArgs, NodeArgs
from dlrover.python.scheduler.kubernetes import (
    _dlrover_context,
    convert_cpu_to_decimal,
    convert_memory_to_mb,
    k8sClient,
)

CONFIGMAP_SUFFIX = "-dlrover-conf"


class VolcanoElasticJob(ElasticJob):
    def __init__(self, job_name, namespace):
        self._namespace = namespace
        self._job_name = job_name

    def get_node_name(self, type, id):
        return "pod-name"

    def get_node_service_addr(self, type, id):
        return ""


class VolcanoJobArgs(JobArgs):
    def __init__(self, platform, namespace, job_name):
        super(VolcanoJobArgs, self).__init__(platform, namespace, job_name)

    def initilize(self):
        self.user = os.getenv("USER", "")
        k8s_client = k8sClient.singleton_instance(self.namespace)

        # Get parameters from configmap
        configmap = self._retry_to_get_configmap(k8s_client)
        self.job_uuid = os.getenv(NodeEnv.JOB_UID, "")
        self.distribution_strategy = configmap.data.get(
            "distributionStrategy", DistributionStrategy.ALLREDUCE
        )
        self.optimizeMode = configmap.data.get(
            "optimizeMode", OptimizeMode.SINGLE_JOB
        )

        # Get parameters from volcano job
        vcjob = self._retry_to_get_vcjob(k8s_client)
        for task in vcjob["spec"]["tasks"]:
            if task["name"] == NodeType.WORKER:
                restart_policy = task["template"]["spec"].get(
                    "restartPolicy", ""
                )
                self.relaunch_always = restart_policy == "Always"

            num = int(task.get("replicas", 0))
            assert len(task["template"]["spec"]["containers"]) == 1
            container = task["template"]["spec"]["containers"][0]
            resources = container.get("resources", {})
            requests = resources.get("requests", {})
            cpu = convert_cpu_to_decimal(requests.get("cpu", 0))
            if "memory" in requests:
                memory = convert_memory_to_mb(requests["memory"])
            else:
                memory = 0
            gpu_type = None
            gpu_num = 0
            for k, v in requests.items():
                if "nvidia.com" in k:
                    gpu_type = k
                    gpu_num = int(v)
            group_resource = NodeGroupResource(
                num,
                NodeResource(
                    cpu=cpu,
                    memory=memory,
                    gpu_type=gpu_type,
                    gpu_num=gpu_num,
                ),
            )
            self.node_args[task["name"]] = NodeArgs(
                group_resource,
                process_timeout=_dlrover_context.seconds_to_timeout_task_process,
            )
        logger.info("Job args = %s", self.__dict__)

    def _retry_to_get_configmap(self, k8s_client: k8sClient):
        for _ in range(3):
            configmap = k8s_client.get_configmap(
                name=self.job_name + CONFIGMAP_SUFFIX,
            )
            if configmap:
                return configmap
            else:
                time.sleep(5)
        raise ValueError("Cannot get the conifgmap %s" % self.job_name)

    def _retry_to_get_vcjob(self, k8s_client: k8sClient):
        for _ in range(3):
            vcjob = k8s_client.get_custom_resource(
                name=self.job_name,
                group="batch.volcano.sh",
                version="v1alpha1",
                plural="jobs",
            )
            if vcjob:
                return vcjob
            else:
                time.sleep(5)
        raise ValueError(
            "Cannot get the training volcano job %s" % self.job_name
        )
