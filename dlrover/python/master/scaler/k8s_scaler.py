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

from dlrover.python.master.resource_generator.base_generator import (
    ResourcePlan,
)
from dlrover.python.master.scaler.base_scaler import Scaler
from dlrover.python.scheduler.kubernetes import Client

SCALER_GROUP = "elastic.iml.github.io"
SCALER_VERION = "v1alpha1"
SCALER_KIND = "Scaler"


class BaseScalerSpec(metaclass=ABCMeta):
    @abstractmethod
    def to_dict(self):
        """Serialize the object to a dictionary"""
        pass


class ScalerResourceSpec(BaseScalerSpec):
    """The resource specification of a node.
    Attributes:
        cpu: CPU cores of a node.
        memory: The memory of of a node with unit of MB.
        gpu: GPU cores of a node.
    """

    def __init__(self, cpu, memory, gpu):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    def to_dict(self):
        spec = {}
        spec["cpu"] = str(self.cpu)
        spec["memory"] = "{}Mi".format(self.memory)
        if self.gpu:
            spec["gpu"] = str(self.gpu)
        return spec


class ScalerReplicaResourceSpec(BaseScalerSpec):
    def __init__(self, replicas, resource_spec) -> None:
        self.replicas = replicas
        self.resource = resource_spec

    def to_dict(self):
        spec = {}
        spec["replicas"] = self.replicas
        spec["resource"] = self.resource.to_dict()
        return spec


class ScalerSpec(BaseScalerSpec):
    def __init__(self, owner_job: str) -> None:
        self.owner_job = owner_job
        self.replica_resource_specs: Dict[str, ScalerReplicaResourceSpec] = {}
        self.node_resource_specs: Dict[str, ScalerResourceSpec] = {}

    def add_replica_resource(self, replica_name, replica_resource_spec):
        self.replica_resource_specs[replica_name] = replica_resource_spec

    def add_node_resource(self, node_name, resource_spec):
        self.node_resource_specs[node_name] = resource_spec

    def to_dict(self):
        spec = {}
        spec["ownerJob"] = self.owner_job
        spec["replicaResourceSpec"] = dict()
        for name, resource_spec in self.replica_resource_specs.items():
            spec["replicaResourceSpec"][name] = resource_spec.to_dict()
        spec["nodeResourceSpec"] = dict()
        for name, resource_spec in self.node_resource_specs.items():
            spec["nodeResourceSpec"] = resource_spec.to_dict()
        return spec


class ScalerKind(BaseScalerSpec):
    """ScalerKind is a dictionary of a Scaler CRD for an ElasticJob
    to scale up/down Pods of a job on a k8s cluster.
    """

    def __init__(
        self,
        api_version: str,
        kind: str,
        metadata: Dict[str, str],
        spec: ScalerSpec,
    ):
        self.api_version = api_version
        self.kind = kind
        self.metadata = metadata
        self.spec = spec

    def to_dict(self):
        spec = {}
        spec["apiVersion"] = self.api_version
        spec["kind"] = self.kind
        spec["metadata"] = self.metadata
        spec["spec"] = self.spec.to_dict()
        return spec


class k8sScaler(Scaler):
    def __init__(self, job_name, namespace, cluster, client: Client):
        super(k8sScaler, self).__init__(job_name)
        self._namespace = namespace
        self._cluster = cluster
        self._client = client

    def scale(self, plan: ResourcePlan):
        scaler_crd = self._generate_scaler_crd_by_plan(plan)
        self._client.create_custom_resource(
            group=SCALER_GROUP,
            version=SCALER_VERION,
            plural="scalers",
            body=scaler_crd.to_dict(),
        )

    def _generate_scaler_crd_by_plan(self, plan: ResourcePlan) -> ScalerKind:
        api_version = SCALER_GROUP + "/" + SCALER_VERION
        scaler_crd = ScalerKind(
            api_version=api_version,
            kind=SCALER_KIND,
            metadata={"name": "{}-scaler".format(self._job_name)},
            spec=ScalerSpec(self._job_name),
        )
        for name, group_resource in plan.task_group_resources.items():
            resource_spec = ScalerResourceSpec(
                cpu=group_resource.node_resource.cpu,
                memory=group_resource.node_resource.memory,
                gpu=group_resource.node_resource.gpu,
            )
            replica_resource_spec = ScalerReplicaResourceSpec(
                replicas=group_resource.count,
                resource_spec=resource_spec,
            )
            scaler_crd.spec.add_replica_resource(name, replica_resource_spec)

        for name, node_resource in plan.node_resources.items():
            resource_spec = ScalerResourceSpec(
                cpu=node_resource.cpu,
                memory=node_resource.memory,
                gpu=node_resource.gpu,
            )
            scaler_crd.spec.add_node_resource(name, resource_spec)
        return scaler_crd
