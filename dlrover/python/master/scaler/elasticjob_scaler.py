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

import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from dlrover.python.common.constants import ElasticJobApi, ScalePlanLabel
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.scheduler.kubernetes import k8sClient


class BaseScaleSpec(metaclass=ABCMeta):
    @abstractmethod
    def to_dict(self):
        """Serialize the object to a dictionary"""
        pass


class ContainerResourceSpec(BaseScaleSpec):
    """The resource specification of a node.
    Attributes:
        cpu: CPU cores of a node.
        memory: The memory of of a node with unit of MB.
        gpu: GPU cores of a node.
    """

    def __init__(self, cpu, memory, gpu_type, gpu_num):
        self.cpu = cpu
        self.memory = memory
        self.gpu_type = gpu_type
        self.gpu_num = gpu_num

    def to_dict(self):
        spec = {}
        spec["cpu"] = str(self.cpu)
        spec["memory"] = "{}Mi".format(self.memory)
        if self.gpu_type:
            spec[self.gpu_type] = str(self.gpu_num)
        return spec


class ReplicaResourceSpec(BaseScaleSpec):
    def __init__(self, replicas, resource_spec: ContainerResourceSpec) -> None:
        self.replicas = replicas
        self.resource = resource_spec

    def to_dict(self):
        spec = {}
        spec["replicas"] = self.replicas
        spec["resource"] = self.resource.to_dict()
        return spec


class PodMeta(BaseScaleSpec):
    def __init__(
        self,
        name,
        id,
        type,
        rank_index,
        service,
        resource: ContainerResourceSpec,
    ):
        self.name = name
        self.id = id
        self.type = type
        self.rank_index = rank_index
        self.service = service
        self.resource = resource

    def to_dict(self):
        spec = {}
        spec["name"] = self.name
        spec["type"] = self.type
        spec["id"] = self.id
        spec["rankIndex"] = self.rank_index
        spec["service"] = self.service
        spec["resource"] = self.resource.to_dict()
        return spec


class ScaleSpec(BaseScaleSpec):
    def __init__(self, owner_job: str) -> None:
        self.owner_job = owner_job
        self.replica_resource_specs: Dict[str, ReplicaResourceSpec] = {}
        self.create_pods: List[PodMeta] = []
        self.remove_pods: List[PodMeta] = []
        self.ps_hosts: List[str] = []
        self.manual_scaling = False

    def to_dict(self):
        spec = {}
        spec["ownerJob"] = self.owner_job
        spec["replicaResourceSpecs"] = dict()
        for name, resource_spec in self.replica_resource_specs.items():
            spec["replicaResourceSpecs"][name] = resource_spec.to_dict()
        spec["createPods"] = []
        for pod in self.create_pods:
            spec["createPods"].append(pod.to_dict())
        spec["removePods"] = []
        for pod in self.remove_pods:
            spec["removePods"].append(pod.to_dict())
        spec["psHosts"] = self.ps_hosts
        return spec


class ScalePlanCrd(BaseScaleSpec):
    """ScalerKind is a dictionary of a Scaler CRD for an ElasticJob
    to scale up/down Pods of a job on a k8s cluster.
    """

    def __init__(
        self,
        api_version: str,
        kind: str,
        metadata,
        spec: ScaleSpec,
    ):
        self.api_version = api_version
        self.kind = kind
        self.metadata = metadata
        self.spec = spec

    def set_owner_reference(self, api_version, name, uid):
        ref_dict = {}
        ref_dict["apiVersion"] = api_version
        ref_dict["blockOwnerDeletion"] = True
        ref_dict["kind"] = ElasticJobApi.ELASTICJOB_KIND
        ref_dict["name"] = name
        ref_dict["uid"] = uid
        self.metadata["ownerReferences"] = [ref_dict]

    def to_dict(self):
        spec = {}
        spec["apiVersion"] = self.api_version
        spec["kind"] = self.kind
        spec["metadata"] = self.metadata
        spec["spec"] = self.spec.to_dict()
        return spec


class ElasticJobScaler(Scaler):
    """ElasticJobScaler creates a elastic.iml.github.io/v1alpha1/Scaler
    CRD to notify ElasticJob controller to scale Pods of a job."""

    def __init__(self, job_name, namespace):
        super(ElasticJobScaler, self).__init__(job_name)
        self._client = k8sClient.singleton_instance(namespace)
        self._namespace = namespace
        self._scaleplan_name = self._job_name + "-scaleplan"
        self._scaleplan_index = 0

    def start(self):
        self._job = self._retry_to_get_job()
        self._job_uid = self._job["metadata"]["uid"]

    def _retry_to_get_job(self):
        for _ in range(3):
            job = self._client.get_custom_resource(
                name=self._job_name,
                group=ElasticJobApi.GROUP,
                version=ElasticJobApi.VERION,
                plural=ElasticJobApi.ELASTICJOB_PLURAL,
            )
            if job:
                return job
            else:
                time.sleep(5)
        raise ValueError("Cannot get the training job %s", self._job_name)

    def scale(self, plan: ScalePlan, **kwargs):
        scale_plan_crd = self._generate_scale_plan_crd(plan)
        self._client.create_custom_resource(
            group=ElasticJobApi.GROUP,
            version=ElasticJobApi.VERION,
            plural=ElasticJobApi.SCALEPLAN_PLURAL,
            body=scale_plan_crd.to_dict(),
        )
        self._scaleplan_index += 1

    def _generate_scale_plan_crd(self, plan: ScalePlan) -> ScalePlanCrd:
        api_version = ElasticJobApi.GROUP + "/" + ElasticJobApi.VERION
        name = self._scaleplan_name + "-" + str(self._scaleplan_index)
        scale_crd = ScalePlanCrd(
            api_version=api_version,
            kind=ElasticJobApi.SCALEPLAN_KIND,
            metadata={"name": name},
            spec=ScaleSpec(self._job_name),
        )
        scale_crd.set_owner_reference(
            api_version, self._job_name, self._job_uid
        )

        for name, group_resource in plan.node_group_resources.items():
            resource_spec = ContainerResourceSpec(
                cpu=group_resource.node_resource.cpu,
                memory=group_resource.node_resource.memory,
                gpu_type=group_resource.node_resource.gpu_type,
                gpu_num=group_resource.node_resource.gpu_num,
            )
            replica_spec = ReplicaResourceSpec(
                replicas=group_resource.count,
                resource_spec=resource_spec,
            )
            scale_crd.spec.replica_resource_specs[name] = replica_spec

        for node in plan.launch_nodes:
            resource_spec = ContainerResourceSpec(
                cpu=node.config_resource.cpu,
                memory=node.config_resource.memory,
                gpu_type=node.config_resource.gpu_type,
                gpu_num=node.config_resource.gpu_num,
            )
            pod_meta = PodMeta(
                node.name,
                node.id,
                node.type,
                node.rank_index,
                node.service_addr,
                resource_spec,
            )
            scale_crd.spec.create_pods.append(pod_meta)

        for node in plan.remove_nodes:
            resource_spec = ContainerResourceSpec(
                cpu=node.config_resource.cpu,
                memory=node.config_resource.memory,
                gpu_type=node.config_resource.gpu_type,
                gpu_num=node.config_resource.gpu_num,
            )
            pod_meta = PodMeta(
                node.name,
                node.id,
                node.type,
                node.rank_index,
                node.service_addr,
                resource_spec,
            )
            scale_crd.spec.remove_pods.append(pod_meta)
        scale_crd.spec.ps_hosts = plan.ps_addrs
        scale_crd.metadata["labels"] = {
            ScalePlanLabel.SCALE_TYPE_KEY: ScalePlanLabel.AUTO_SCALE
        }
        return scale_crd
