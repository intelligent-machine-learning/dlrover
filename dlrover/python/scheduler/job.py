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
from typing import Dict, Optional

from dlrover.python.common.constants import (
    Accelerators,
    DistributionStrategy,
    NodeType,
)
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.common.serialize import JsonSerializable


class ElasticJob(metaclass=ABCMeta):
    def __init__(self, namespace, job_name):
        """
        ElasticJob manages Pods by K8s Python APIs. The example of an elastic
        job is in dlrover/go/elasticjob_operator/config/samples/
        elastic_v1alpha1_elasticjob.yaml
        Args:
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
        restart_timeout=0,
        critical_nodes="",
        process_timeout=1800,
    ):
        self.group_resource = group_resource
        self.restart_count = restart_count
        self.auto_scale = auto_scale
        self.restart_timeout = restart_timeout
        self.critical_nodes = critical_nodes
        self.process_timeout = process_timeout


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
            configurations of nodes.
        enable_dynamic_sharding: Whether to use dynamic sharding of a dataset.
        enable_elastic_scheduling: Whether to use elastic scheduling.
        distribution_strategy: The distributed training strategy supports
            ["ParameterServerStrategy", "AllReduceStrategy"]
        relaunch_always (bool): whether to always relaunch the error node.
        remove_exited_node (bool): whether to remove the exited node.
        cordon_fault_node (bool): whether to set the fault node unscheduled.
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
        self.resource_limits = ResourceLimits()
        self.relaunch_always = True
        self.remove_exited_node = False
        self.cordon_fault_node = False
        self.xpu_type: Accelerators = Accelerators.GENERIC_CPU
        self.enable_suspended = False
        self.training_elastic_mode = "base"
        self.group_affinity: Optional[Dict[int, int]] = None
        # Node-group scheduling strategy for worker nodes; None (default)
        # when the job configures no node-group affinity. Configuring
        # group_affinity or soft_group_affinity requires the strategy to be
        # set explicitly, otherwise the master fails to start. See
        # NodeGroupStrategy in dlrover.python.common.constants and
        # NodeGroupSchedule/resolve_group_id in
        # dlrover.python.master.resource.job.
        self.node_group_strategy: Optional[str] = None
        # Megatron parallel sizes, consulted when a node-group affinity is
        # configured (validated by validate_topology in
        # master.resource.job and validate_soft_group_topology in
        # master.resource.soft_group). The real EP group size is
        # expert_tensor_parallel_size * expert_model_parallel_size, where
        # an unset expert_tensor_parallel_size (None) inherits
        # tensor_model_parallel_size at the master layer. TP and CP are
        # recorded but not used by the scheduling today.
        self.tensor_model_parallel_size: int = 1
        self.pipeline_model_parallel_size: int = 1
        self.expert_model_parallel_size: int = 1
        self.expert_tensor_parallel_size: Optional[int] = None
        self.context_parallel_size: int = 1
        # Unequal-size node group pod counts (--soft-group-affinity),
        # fully isolated from group_affinity above and mutually exclusive
        # with it. See SoftGroupSchedule / validate_soft_group_topology /
        # resolve_soft_group_id in master.resource.soft_group. Every group
        # size must be a multiple of the EP slot (ETP*EP/R pods) — the EP
        # alignment is mandatory, not an option.
        self.soft_group_affinity: Optional[Dict[int, int]] = None
        # With soft_group_affinity: relaunch (FO) workers without their
        # node-group labels so they can be scheduled onto any segment.
        self.no_group_failover: bool = False
        # Reorder the elastic-training rendezvous world by the psw
        # topology observed at rendezvous time (GroupTopologyQuerier +
        # GroupTopologySorter in
        # dlrover.python.master.elastic_training.topology_rerank).
        # Requires node_group_strategy; mutually exclusive with
        # group_affinity and soft_group_affinity.
        self.enable_topology_rerank: bool = False

    @abstractmethod
    def initilize(self):
        pass


class LocalJobArgs(JobArgs):
    def __init__(self, platform, namespace, job_name):
        super().__init__(platform, namespace, job_name)

    def initilize(self):
        self.distribution_strategy = DistributionStrategy.LOCAL
        self.enable_elastic_scheduling = False
        worker_group = NodeGroupResource(1, NodeResource(0, 0))
        self.node_args[NodeType.WORKER] = NodeArgs(worker_group)
