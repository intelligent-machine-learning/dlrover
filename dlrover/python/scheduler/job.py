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

from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.constants import DistributionStrategy
from dlrover.python.common.node import NodeGroupResource
from dlrover.python.common.serialize import JsonSerializable


class ElasticJob(metaclass=ABCMeta):
    def __init__(self, namespace, job_name):
        """
        ElasticJob manages Pods by K8s Python APIs. The example of an elastic
        job is in dlrover/go/elasticjob_operator/config/samples/
        elastic_v1alpha1_elasticjob.yaml
        Args:
            image_name: Docker image path for DLRover pod.
            namespace: The name of the Kubernetes namespace where DLRover
                pods will be created.
            job_name: DLRover job name, should be unique in the namespace.
                Used as pod name prefix and value for "elastic" label.
        """
        self._namespace = namespace
        self._job_name = job_name

    @abstractmethod
    def get_node_service_addr(self, type, id):
        pass

    @abstractmethod
    def get_node_name(self, type, id):
        pass


class NodeArgs(metaclass=ABCMeta):
    def __init__(
        self,
        group_resource: NodeGroupResource,
        auto_scale=True,
        restart_count=1,
        retart_timeout=0,
        critical_nodes="",
    ):
        self.group_resource = group_resource
        self.restart_count = restart_count
        self.auto_scale = auto_scale
        self.restart_timeout = retart_timeout
        self.critical_nodes = critical_nodes


class ResourceLimits(object):
    def __init__(self, cpu=0, memory=0, gpu_num=0) -> None:
        self.cpu = cpu
        self.memory = memory
        self.gpu_num = gpu_num


class JobArgs(JsonSerializable):
    """JobArgs are arguments of an elastic training job.
    Attributes:
        namespace: The name of the Kubernetes namespace where the
            job is created.
        job_name: The name of the job.
        node_args: It contains resource and elasticity
            configuraions of nodes.
        enable_dynamic_sharding: Whether to use dynamic sharding of a dataset.
        enable_elastic_scheduling: Whether to use elastic scheduling.
        distribution_strategy:
    """

    def __init__(self, platform, namespace, job_name):
        self.platform = platform
        self.namespace = namespace
        self.job_name = job_name
        self.node_args: Dict[str, NodeArgs] = {}
        self.enable_dynamic_sharding = True
        self.enable_elastic_scheduling = True
        self.distribution_strategy = DistributionStrategy.PS
        self.job_uuid = ""
        self.user = ""
        self.cluster = ""
        self.optimize_mode = "single-job"
        self.use_ddp = False
        self.resource_limits = ResourceLimits()

    @abstractmethod
    def initilize(self):
        pass
