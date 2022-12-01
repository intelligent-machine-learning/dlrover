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
import time
from typing import Dict, List

from kubernetes import client
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource
from dlrover.python.master.scaler.base_scaler import Scaler, ScalePlan, LaunchNode
from dlrover.python.scheduler.kubernetes import k8sClient, NODE_SERVICE_PORTS, get_pod_name
from dlrover.python.common.constants import (
    ElasticJobLabel,
    NodeType,
    NodeStatus,
    DistributionStrategy,
)

ELASTICDL_APP_NAME = "elastic"
ELASTICDL_JOB_KEY = "elastic-job-name"
ELASTICDL_REPLICA_TYPE_KEY = "elastic-replica-type"
ELASTICDL_REPLICA_INDEX_KEY = "elastic-replica-index"


class PodEnv(object):
    RELAUNCHED_POD = "RELAUNCHED_POD"
    ELASTICDL_ENABLED = "ELASTICDL_ENABLED"
    MASTER_ADDR = "MASTER_ADDR"
    WORKER_TYPE = "WORKER_TYPE"
    WORKER_ID = "WORKER_ID"
    WORKER_NUM = "WORKER_NUM"


def append_pod_ip_to_env(env):
    pod_ip_var = V1EnvVar(
        name="MY_POD_IP",
        value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.podIP")
        ),
    )
    node_ip_var = V1EnvVar(
        name="MY_NODE_IP",
        value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.hostIP")
        ),
    )
    if env:
        env.append(pod_ip_var)
        env.append(node_ip_var)
    else:
        env = [pod_ip_var, node_ip_var]
    return env


class BashCommandTemplate(object):
    REDIRECTION = "({}) 2>&1 | tee {}"
    SET_PIPEFAIL = "set -o pipefail;"


class PodTemplate(object):
    """PodTemplate is the template of replica in a job"""

    def __init__(self, template):
        self.restart_policy = template["spec"]["restartPolicy"]
        main_container = template["spec"]["containers"][0]
        self.name = main_container["name"]
        self.image = main_container["image"]
        self.command = main_container["command"]
        self.image_pull_policy = main_container.get(
            "imagePullPolicy", "Always"
        )


class PodScaler(Scaler):
    """PodScaler launches or removes Pods using Kubernetes Python APIs.
    """
    def __init__(self, job_name, namespace, cluster):
        super(PodScaler, self).__init__(job_name)
        self._k8s_client = k8sClient(namespace, job_name)
        self._namespace = namespace
        self._cluster = cluster
        self._replica_template: Dict[str, PodTemplate] = {}
        job = self._retry_to_get_job()
        self._distribution_strategy = job["spec"]["distributionStrategy"]
        for replica, spec in job["spec"]["replicaSpecs"].items():
            self._replica_template[replica] = PodTemplate(spec["template"])

    def _retry_to_get_job(self):
        for _ in range(3):
            job = self._k8s_client.get_training_job()
            if job:
                return job
            else:
                time.sleep(5)
        raise ValueError("Cannot get the training job %s", self._job_name)

    def scale(self, plan: ScalePlan):
        job_pods = self._stats_alive_pods()

        if not plan.launch_nodes and not plan.remove_nodes:
            for type, group_resource in plan.node_group_resources.items():
                cur_pods = job_pods[type]
                if group_resource.count > len(cur_pods):
                    self._scale_up_pods()
                elif group_resource.count < len(cur_pods):
                    self._scale_down_pods()
        for node in plan.launch_nodes:
            self._create_typed_pod(
                plan.node_group_resources, node, plan.ps_addrs
            )
        for pod in plan.remove_nodes:
            self._k8s_client.delete_pod(pod)

    def _stats_alive_pods(self):
        job_selector = ElasticJobLabel.JOB_KEY + "=" + self._job_name
        pod_list = self._k8s_client.list_namespaced_pod(job_selector)
        job_pods = {}
        for pod in pod_list.items:
            pod_type = pod.metadata.labels[ElasticJobLabel.REPLICA_TYPE_KEY]
            job_pods.setdefault(pod_type, [])
            pod_id = int(
                pod.metadata.labels[ElasticJobLabel.REPLICA_INDEX_KEY]
            )
            task_id = int(
                pod.metadata.labels[ElasticJobLabel.TRAINING_TASK_INDEX_KEY]
            )
            node = Node(
                node_type=pod_type,
                node_id=pod_id,
                name=pod.metadata.name,
                task_index=task_id,
                status=pod.status.phase,
                config_resource=None,
            )
            if node.status in [NodeStatus.PENDING, NodeStatus.RUNNING]:
                job_pods[pod_type].append(node)

    def _scale_up_pods(self):
        pass

    def _scale_down_pods(self):
        pass

    def _get_common_labels(self):
        """Labels that should be attached to all k8s objects belong to
        current job.
        """
        return {"app": ELASTICDL_APP_NAME, ELASTICDL_JOB_KEY: self._job_name}

    def get_node_service_addr(self, type, id):
        service_name = get_pod_name(self._job_name, type, id)
        return "%s.%s.svc:%d" % (
            service_name,
            self._namespace,
            NODE_SERVICE_PORTS[type],
        )

    def get_master_pod(self):
        master_pod_name = "elastic-%s-master" % self._job_name
        return self._k8s_client.get_pod(master_pod_name)

    def get_typed_pod(self, pod_type, id):
        pod_name = get_pod_name(self._job_name, pod_type, id)
        return self._k8s_client.get_pod(pod_name)

    def _create_typed_pod(self, job_resource,  pod: LaunchNode, ps_addrs):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        pod_name = get_pod_name(self._job_name, pod.type, pod.node_id)
        master_pod = self.get_master_pod()
        env: List[V1EnvVar] = []
        env = append_pod_ip_to_env(env)

        env.append(V1EnvVar(name=PodEnv.WORKER_TYPE, value=pod.type))

        if self._distribution_strategy == DistributionStrategy.PARAMETER_SERVER and ps_addrs:
            tf_config = new_tf_config(
                job_resource,
                self.get_node_service_addr,
                pod.type,
                pod.task_id,
                ps_addrs,
            )
            if tf_config:
                env.append(
                    V1EnvVar(name="TF_CONFIG", value=json.dumps(tf_config))
                )

        lifecycle = None

        if pod.type not in self._replica_template:
            raise ValueError(
                "No replica %s specification in job %s",
                pod.type,
                self._job_name,
            )
        pod_template = self._replica_template[pod.type]

        labels = self._get_common_labels()

        pod = self.create_pod_obj(
            name=pod_name,
            image=pod_template.image,
            command=pod_template.command,
            resource_requests=pod.resource.to_resource_dict(),
            resource_limits=pod.resource.to_resource_dict(),
            priority=pod.resource.priority,
            image_pull_policy=pod_template.image_pull_policy,
            restart_policy=pod_template.restart_policy,
            owner=master_pod,
            env=env,
            lifecycle=lifecycle,
            labels=labels,
        )
        # Add replica type and index
        pod.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY] = pod.type
        pod.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY] = str(pod.node_id)
        self._k8s_client.create_pod(pod)
        return pod

    def delete_typed_pod(self, pod_type, id):
        pod_name = self.get_node_name(pod_type, id)
        self._k8s_client.delete_pod(pod_name)

    def create_service(self, pod_type, pod_id, service_name):
        # Use master pod as service owner so the service will not be
        #  deleted if the corresponding pod is deleted.
        service = self._create_service_obj(
            name=service_name,
            port=NODE_SERVICE_PORTS[pod_type],
            target_port=NODE_SERVICE_PORTS[pod_type],
            replica_type=pod_type,
            replica_index=pod_id,
            owner=self.get_master_pod(),
        )
        self._k8s_client.create_service(service)
        return service

    def create_service_with_retry(
        self, pod_type, pod_id, service_name, retry_num=5
    ):
        for _ in range(retry_num):
            succeed = self.create_service(pod_type, pod_id, service_name)
            if succeed:
                return succeed
            else:
                time.sleep(5)
        return False

    def patch_service(self, pod_type, id, service_name):
        service = self._create_service_obj(
            name=service_name,
            port=NODE_SERVICE_PORTS[pod_type],
            target_port=NODE_SERVICE_PORTS[pod_type],
            replica_type=pod_type,
            replica_index=id,
            owner=self.get_master_pod(),
        )
        self._k8s_client.patch_service(service_name, service)

    def patch_service_with_retry(
        self, pod_type, new_id, service_name, retry_num=5
    ):
        for _ in range(retry_num):
            succeed = self.patch_service(pod_type, new_id, service_name)
            if succeed:
                return succeed
            else:
                time.sleep(5)
        return False

    def _create_service_obj(
        self,
        name,
        port,
        target_port,
        replica_type,
        replica_index,
        owner=None,
    ):
        labels = self._get_common_labels()

        metadata = client.V1ObjectMeta(
            name=name,
            labels=labels,
            # Note: We have to add at least one annotation here.
            # Otherwise annotation is `None` and cannot be modified
            # using `with_service()` for cluster specific information.
            annotations=labels,
            owner_references=self.create_owner_reference(owner)
            if owner
            else None,
            namespace=self._namespace,
        )
        selector = {
            "app": ELASTICDL_APP_NAME,
            ELASTICDL_JOB_KEY: self._job_name,
            ELASTICDL_REPLICA_TYPE_KEY: replica_type,
            ELASTICDL_REPLICA_INDEX_KEY: str(replica_index),
        }
        spec = client.V1ServiceSpec(
            ports=[client.V1ServicePort(port=port, target_port=target_port)],
            selector=selector,
            type=None,
        )
        service = client.V1Service(
            api_version="v1", kind="Service", metadata=metadata, spec=spec
        )
        return service

    def create_persistent_volume_claim(
        self, pvc_name, storage, iolimits_enabled
    ):
        # Use master pod as service owner so the service will not be
        #  deleted if the corresponding pod is deleted.
        pvc = self._create_persistent_volume_claim_object(
            name=pvc_name,
            storage=storage,
            owner=self.get_master_pod(),
            iolimits_enabled=iolimits_enabled,
        )
        return self._k8s_client.create_pvc(pvc)

    def _create_persistent_volume_claim_object(
        self, name, storage, owner, iolimits_enabled
    ):
        labels = self._get_common_labels()
        # Support speed limitation of SSD
        annotations = {}
        if iolimits_enabled:
            annotations["alibabacloud.csi.alipay.com/enable-iolimits"] = "true"
        metadata = client.V1ObjectMeta(
            name=name,
            labels=labels,
            # Note: We have to add at least one annotation here.
            # Otherwise annotation is `None` and cannot be modified
            # using `with_service()` for cluster specific information.
            annotations=annotations,
            owner_references=self.create_owner_reference(owner)
            if owner
            else None,
            namespace=self._namespace,
        )

        spec = client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(
                requests={"storage": storage}
            ),
            storage_class_name="alibabacloud-raw-file",
            volume_mode="Filesystem",
        )
        pvc = client.V1PersistentVolumeClaim(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata=metadata,
            spec=spec,
        )
        return pvc

    def create_pod_obj(
        self,
        name,
        owner,
        image,
        command,
        resource_requests: Dict[str, float],
        resource_limits: Dict[str, float],
        image_pull_policy,
        lifecycle,
        env,
        restart_policy,
        priority,
        labels,
        termination_period=None,
    ):
        resource_limits = (
            resource_limits if len(resource_limits) > 0 else resource_requests
        )
        container = client.V1Container(
            name="main",
            image=image,
            command=command,
            resources=client.V1ResourceRequirements(
                requests=resource_requests,
                limits=resource_limits,
            ),
            image_pull_policy=image_pull_policy,
            env=env,
            lifecycle=lifecycle,
        )

        # Pod
        spec = client.V1PodSpec(
            containers=[container],
            restart_policy=restart_policy,
            priority_class_name=priority,
            termination_grace_period_seconds=termination_period,
        )

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=name,
                labels=labels,
                owner_references=self.create_owner_reference(owner),
                namespace=self._namespace,
            ),
        )
        return pod

    @staticmethod
    def create_owner_reference(owner_pod):
        owner_ref = (
            [
                client.V1OwnerReference(
                    api_version="v1",
                    block_owner_deletion=True,
                    kind="Pod",
                    name=owner_pod.metadata.name,
                    uid=owner_pod.metadata.uid,
                )
            ]
            if owner_pod
            else None
        )
        return owner_ref


def new_tf_config(
    job_resource: Dict[str, NodeGroupResource],
    new_service_fn,
    type_key,
    index_key,
    ps_addrs,
):
    """Get tf.estimator config spec data. The detail is in
    https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
    """
    cluster_dict = {}
    cluster_dict[NodeType.PS] = ps_addrs
    worker_num = job_resource[NodeType.WORKER].count
    if type_key == NodeType.WORKER and index_key >= worker_num:
        worker_num = index_key + 1
    if worker_num > 0:
        cluster_dict[NodeType.WORKER] = []
        for worker_id in range(worker_num):
            cluster_dict[NodeType.WORKER].append(
                new_service_fn(NodeType.WORKER, worker_id)
            )
    evaluator_num = job_resource[NodeType.EVALUATOR].count
    if evaluator_num > 0:
        cluster_dict[NodeType.EVALUATOR] = []
        for worker_id in range(evaluator_num):
            cluster_dict[NodeType.EVALUATOR].append(
                new_service_fn(NodeType.EVALUATOR, worker_id)
            )
    chief_num = job_resource[NodeType.CHIEF].count
    if chief_num > 0:
        cluster_dict[NodeType.CHIEF] = []
        for worker_id in range(chief_num):
            cluster_dict[NodeType.CHIEF].append(
                new_service_fn(NodeType.CHIEF, worker_id)
            )

    task_dict = {}
    task_dict["type"] = type_key
    task_dict["index"] = index_key
    return {"cluster": cluster_dict, "task": task_dict}
