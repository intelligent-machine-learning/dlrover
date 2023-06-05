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
import json
import threading
import time
from typing import Dict, List

from kubernetes import client
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector

from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    NodeEnv,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.scheduler.kubernetes import (
    NODE_SERVICE_PORTS,
    get_pod_name,
    k8sClient,
)


class FakeKubeResponse:
    def __init__(self, obj):
        self.data = json.dumps(obj)


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


class PodScaler(Scaler):
    """PodScaler launches or removes Pods using Kubernetes Python APIs
    by a ScalePlan. After PodScaler receives a ScalePlan, it will
    list all alive Pods of the job and push new `Node`s to a queue.
    The thread of creating Pods will create Pods if there is `Node`
    in a queue.
    """

    def __init__(self, job_name, namespace):
        super(PodScaler, self).__init__(job_name)
        self._k8s_client = k8sClient.singleton_instance(namespace)
        self._namespace = namespace
        self._replica_template: Dict[str, client.V1Pod] = {}
        self._create_node_queue: List[Node] = []
        self._lock = threading.Lock()
        self._plan = ScalePlan()
        self._pod_stats: Dict[str, int] = {}
        self._init_pod_config_by_job()
        threading.Thread(
            target=self._periodic_create_pod, name="pod-creater", daemon=True
        ).start()
        self.api_client = client.ApiClient()
        self._k8s_client.api_instance

    def _init_pod_config_by_job(self):
        self._job = self._retry_to_get_job()
        self._distribution_strategy = self._job["spec"].get(
            "distributionStrategy", None
        )
        worker_spec = self._job["spec"]["replicaSpecs"][NodeType.WORKER]
        self._config_worker_num = worker_spec.get("replicas", 0)
        if "replicaSpecs" in self._job["spec"]:
            for replica, spec in self._job["spec"]["replicaSpecs"].items():
                if replica == NodeType.DLROVER_MASTER:
                    continue
                pod = spec["template"]
                pod["apiVesion"] = "v1"
                pod["kind"] = "Pod"
                res = FakeKubeResponse(pod)
                v1pod = self._k8s_client.api_client.deserialize(res, "V1Pod")
                self._replica_template[replica] = v1pod

    def _retry_to_get_job(self):
        for _ in range(3):
            job = self._k8s_client.get_custom_resource(
                name=self._job_name,
                group="elastic.iml.github.io",
                version="v1alpha1",
                plural="elasticjobs",
            )
            if job:
                return job
            else:
                time.sleep(5)
        raise ValueError("Cannot get the training job %s", self._job_name)

    def scale(self, plan: ScalePlan):
        """Scale in/out Pods by a ScalePlan."""
        self._plan = plan
        job_pods = self._list_job_pods()
        logger.info("Scale the job by plan %s", plan.toJSON())

        with self._lock:
            for type, group_resource in plan.node_group_resources.items():
                type_pods = job_pods.get(type, [])
                max_pod_id = self._get_max_pod_id(type_pods)
                normal_pods = []
                for node in type_pods:
                    if node.status in [
                        NodeStatus.PENDING,
                        NodeStatus.RUNNING,
                        NodeStatus.SUCCEEDED,
                    ]:
                        normal_pods.append(node)
                cur_pods = normal_pods + self._get_type_pod_in_queue(type)
                if group_resource.count > len(cur_pods):
                    self._scale_up_pods(type, plan, cur_pods, max_pod_id)
                elif group_resource.count < len(cur_pods):
                    self._scale_down_pods(type, plan, cur_pods)
            for node in plan.launch_nodes:
                self._create_node_queue.append(node)
            for node in plan.remove_nodes:
                removed = self._remove_not_create_pod(node.name)
                if not removed:
                    self._k8s_client.delete_pod(node.name)
            self._update_job_pods(job_pods)

    def _update_job_pods(self, job_pods: Dict[str, List[Node]]):
        for type in [
            NodeType.CHIEF,
            NodeType.MASTER,
            NodeType.PS,
            NodeType.WORKER,
            NodeType.EVALUATOR,
        ]:
            cur_pods = job_pods.get(type, []) + self._get_type_pod_in_queue(
                type
            )
            self._pod_stats[type] = len(cur_pods)

    def _get_type_pod_in_queue(self, type):
        pods = []
        for pod in self._create_node_queue:
            if pod.type == type:
                pods.append(pod)
        return pods

    def _remove_not_create_pod(self, pod_name):
        not_created_pod = None
        for pod in self._create_node_queue:
            if pod_name == get_pod_name(self._job_name, pod.type, pod.id):
                not_created_pod = pod
                break
        if not_created_pod:
            self._create_node_queue.remove(not_created_pod)
            return True
        return False

    def _list_job_pods(self):
        job_selector = ElasticJobLabel.JOB_KEY + "=" + self._job_name
        pod_list = self._k8s_client.list_namespaced_pod(job_selector)
        job_pods: Dict[str, List[Node]] = {}
        for pod in pod_list.items:
            pod_type = pod.metadata.labels[ElasticJobLabel.REPLICA_TYPE_KEY]
            if pod_type == NodeType.DLROVER_MASTER:
                continue
            job_pods.setdefault(pod_type, [])
            pod_id = int(
                pod.metadata.labels[ElasticJobLabel.REPLICA_INDEX_KEY]
            )
            task_id = int(pod.metadata.labels[ElasticJobLabel.RANK_INDEX_KEY])
            pod_resource = self._get_pod_resource(pod)
            node = Node(
                node_type=pod_type,
                node_id=pod_id,
                name=pod.metadata.name,
                rank_index=task_id,
                status=pod.status.phase,
                config_resource=pod_resource,
            )
            job_pods[pod_type].append(node)
        return job_pods

    def _get_pod_resource(self, pod):
        resources = pod.spec.containers[0].resources
        cpu = NodeResource.convert_cpu_to_decimal(
            resources.requests.get("cpu", "0")
        )
        if "memory" in resources.requests:
            memory = NodeResource.convert_memory_to_mb(
                resources.requests["memory"]
            )
        else:
            memory = 0
        return NodeResource(cpu, memory)

    def _scale_up_pods(
        self,
        type,
        plan: ScalePlan,
        cur_pods: List[Node],
        max_pod_id,
    ):
        """The method will create a Node instances and push it into a queue.
        The thread to create Pods will periodicall create Pods by
        the Node instance in the queue."""
        cur_num = len(cur_pods)
        group_resource = plan.node_group_resources[type]
        up_num = group_resource.count - cur_num
        for i in range(up_num):
            node_id = max_pod_id + 1 + i
            task_id = cur_num + i
            node = Node(
                type,
                node_id,
                copy.deepcopy(group_resource.node_resource),
                rank_index=task_id,
                name=get_pod_name(self._job_name, type, node_id),
                service_addr=self.get_node_service_addr(type, task_id),
            )
            self._create_node_queue.append(node)

    def _get_max_pod_id(self, pods: List[Node]):
        max_id = -1
        for pod in pods:
            max_id = max(pod.id, max_id)
        return max_id

    def _scale_down_pods(
        self,
        type,
        plan: ScalePlan,
        cur_pods: List[Node],
    ):
        """Delete Pods to scale down Pods."""
        group_resource = plan.node_group_resources[type]
        down_num = len(cur_pods) - group_resource.count

        not_created_pods = []
        for pending_pod in self._create_node_queue:
            if pending_pod.type == type:
                not_created_pods.append(pending_pod)
        while down_num > 0 and not_created_pods:
            pod = not_created_pods.pop()
            self._create_node_queue.remove(pod)
            down_num -= 1
        cur_pods.sort(key=lambda x: x.id, reverse=True)
        for pod in cur_pods:
            if down_num <= 0:
                break
            self._k8s_client.delete_pod(pod.name)
            down_num -= 1

    def _get_common_labels(self):
        """Labels that should be attached to all k8s objects belong to
        current job.
        """
        return {
            "app": ElasticJobLabel.APP_NAME,
            ElasticJobLabel.JOB_KEY: self._job_name,
        }

    def get_node_service_addr(self, type, id):
        service_name = get_pod_name(self._job_name, type, id)
        return "%s.%s.svc:%d" % (
            service_name,
            self._namespace,
            NODE_SERVICE_PORTS[type],
        )

    def get_typed_pod(self, pod_type, id):
        pod_name = get_pod_name(self._job_name, pod_type, id)
        return self._k8s_client.get_pod(pod_name)

    def _periodic_create_pod(self):
        while True:
            with self._lock:
                while self._create_node_queue:
                    node = self._create_node_queue.pop(0)
                    succeed = False
                    if self._check_cluster_ready_for_pod(node):
                        pod = self._create_pod(
                            node,
                            self._pod_stats,
                            self._plan.ps_addrs,
                        )
                        succeed = self._k8s_client.create_pod(pod)
                    if not succeed:
                        self._create_node_queue.insert(0, node)
                        break
                    service_ready = self._create_service_for_pod(node)
                    if not service_ready:
                        self._create_node_queue.insert(0, node)
                        break
            time.sleep(3)

    def _check_cluster_ready_for_pod(self, node: Node):
        """Check whether the resource of a cluster is enough to
        create a node"""
        return True

    def _create_pod(self, node: Node, pod_stats: Dict[str, int], ps_addrs):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        node.update_priority(pod_stats[node.type])
        pod_name = get_pod_name(self._job_name, node.type, node.id)
        logger.info(
            "Create Pod %s with resource %s",
            pod_name,
            node.config_resource.to_resource_dict(),
        )
        env: List[V1EnvVar] = []
        env = append_pod_ip_to_env(env)

        env.append(V1EnvVar(name=NodeEnv.WORKER_TYPE, value=node.type))
        env.append(V1EnvVar(name=NodeEnv.WORKER_ID, value=str(node.id)))

        # A deadlock can happen when pthread_atfork handler is running.
        # For detail https://chromium.googlesource.com/external/github.com/grpc/grpc/+/refs/tags/v1.19.0-pre1/doc/fork_support.md  # noqa: E501
        env.append(V1EnvVar(name=NodeEnv.GRPC_ENABLE_FORK, value="False"))

        worker_num = self._config_worker_num
        if worker_num == 0:
            worker_num = pod_stats[node.type]
        env.append(V1EnvVar(name=NodeEnv.WORKER_NUM, value=str(worker_num)))
        env.append(
            V1EnvVar(name=NodeEnv.WORKER_RANK, value=str(node.rank_index))
        )
        master_service = "elasticjob-{}-dlrover-master:50001".format(
            self._job_name
        )
        env.append(
            V1EnvVar(name=NodeEnv.DLROVER_MASTER_ADDR, value=master_service)
        )
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            torch_master_ip = self.get_first_worker_host()
            if torch_master_ip:
                env.append(
                    V1EnvVar(name=NodeEnv.RDZV_ENDPOINT, value=torch_master_ip)
                )
            else:
                node_ip_var = V1EnvVar(
                    name=NodeEnv.RDZV_ENDPOINT,
                    value_from=V1EnvVarSource(
                        field_ref=V1ObjectFieldSelector(
                            field_path="status.podIP"
                        )
                    ),
                )
                env.append(node_ip_var)

        env.append(
            V1EnvVar(
                name=NodeEnv.POD_NAME,
                value_from=V1EnvVarSource(
                    field_ref=V1ObjectFieldSelector(field_path="metadata.name")
                ),
            )
        )

        node_type = node.type
        if node.type not in self._replica_template:
            if node.type in [NodeType.CHIEF, NodeType.EVALUATOR]:
                node_type = NodeType.WORKER
        if node_type not in self._replica_template:
            raise ValueError(
                "No replica %s specification in job %s",
                node.type,
                self._job_name,
            )
        pod_template = self._replica_template[node_type]
        labels = self._get_common_labels()
        pod = self._create_pod_obj(
            name=pod_name,
            pod_template=pod_template,
            resource_requests=node.config_resource.to_resource_dict(),
            resource_limits=node.config_resource.to_resource_dict(),
            priority=node.config_resource.priority,
            env=env,
            lifecycle=None,
            labels=labels,
        )
        # Add replica type and index
        pod.metadata.labels[ElasticJobLabel.REPLICA_TYPE_KEY] = node.type
        pod.metadata.labels[ElasticJobLabel.REPLICA_INDEX_KEY] = str(node.id)
        pod.metadata.labels[ElasticJobLabel.RANK_INDEX_KEY] = str(
            node.rank_index
        )
        self._patch_tf_config_into_env(pod, node, pod_stats, ps_addrs)
        return pod

    def get_first_worker_host(self):
        pod = self.get_typed_pod(NodeType.WORKER, 0)
        if pod:
            return pod.status.pod_ip
        return ""

    def _patch_tf_config_into_env(self, pod, node: Node, pod_stats, ps_addrs):
        if self._distribution_strategy == DistributionStrategy.PS and ps_addrs:
            tf_config = new_tf_config(
                pod_stats,
                self.get_node_service_addr,
                node.type,
                node.rank_index,
                ps_addrs,
            )
            if tf_config:
                pod.spec.containers[0].env.append(
                    V1EnvVar(name="TF_CONFIG", value=json.dumps(tf_config))
                )

    def _delete_typed_pod(self, pod_type, id):
        pod_name = get_pod_name(self._job_name, pod_type, id)
        self._k8s_client.delete_pod(pod_name)

    def _create_service_for_pod(self, node: Node):
        # create or patch worker service
        service_ready = True
        if node.service_addr:
            service_name = node.service_addr.split(".")[0]
        else:
            service_name = get_pod_name(
                self._job_name, node.type, node.rank_index
            )
        if not self._k8s_client.get_service(service_name):
            succeed = self._create_service_with_retry(
                node.type, node.rank_index, service_name
            )
            service_ready = service_ready and succeed
        else:
            succeed = self._patch_service_with_retry(
                node.type, node.rank_index, service_name
            )
            service_ready = service_ready and succeed
        if not service_ready:
            logger.error(
                "Fail to create service %s for the %s pod %s",
                service_name,
                node.type,
                node.id,
            )
            self._delete_typed_pod(node.type, node.id)
            service_ready = False
        return service_ready

    def _create_service_with_retry(
        self, pod_type, rank_id, service_name, retry_num=5
    ):
        for _ in range(retry_num):
            succeed = self._create_service(pod_type, rank_id, service_name)
            if succeed:
                return succeed
            else:
                time.sleep(5)
        return False

    def _create_service(self, pod_type, rank_id, service_name):
        # Use master pod as service owner so the service will not be
        #  deleted if the corresponding pod is deleted.
        service = self._create_service_obj(
            name=service_name,
            port=NODE_SERVICE_PORTS[pod_type],
            target_port=NODE_SERVICE_PORTS[pod_type],
            replica_type=pod_type,
            rank_index=rank_id,
        )

        return self._k8s_client.create_service(service)

    def _patch_service_with_retry(
        self, pod_type, new_id, service_name, retry_num=5
    ):
        for _ in range(retry_num):
            succeed = self._patch_service(pod_type, new_id, service_name)
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
        rank_index,
    ):
        labels = self._get_common_labels()

        metadata = client.V1ObjectMeta(
            name=name,
            labels=labels,
            # Note: We have to add at least one annotation here.
            # Otherwise annotation is `None` and cannot be modified
            # using `with_service()` for cluster specific information.
            annotations=labels,
            owner_references=[self._create_job_owner_reference()],
            namespace=self._namespace,
        )
        selector = {
            ElasticJobLabel.JOB_KEY: self._job_name,
            ElasticJobLabel.REPLICA_TYPE_KEY: replica_type,
            ElasticJobLabel.RANK_INDEX_KEY: str(rank_index),
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

    def _patch_service(self, pod_type, rank_id, service_name):
        service = self._create_service_obj(
            name=service_name,
            port=NODE_SERVICE_PORTS[pod_type],
            target_port=NODE_SERVICE_PORTS[pod_type],
            replica_type=pod_type,
            rank_index=rank_id,
        )
        return self._k8s_client.patch_service(service_name, service)

    def _create_pod_obj(
        self,
        name,
        pod_template: client.V1Pod,
        resource_requests: Dict[str, float],
        resource_limits: Dict[str, float],
        lifecycle,
        env,
        priority,
        labels,
        termination_period=None,
    ):
        pod = copy.deepcopy(pod_template)
        main_container: client.V1Container = pod.spec.containers[0]
        resource_limits = (
            resource_limits if len(resource_limits) > 0 else resource_requests
        )
        main_container.resources = client.V1ResourceRequirements(
            requests=resource_requests,
            limits=resource_limits,
        )
        main_container.env = env
        main_container.lifecycle = lifecycle
        pod.spec.priority_class_name = priority
        pod.spec.termination_grace_period_seconds = termination_period
        pod.metadata = client.V1ObjectMeta(
            name=name,
            labels=labels,
            owner_references=[self._create_job_owner_reference()],
            namespace=self._namespace,
        )
        return pod

    def _create_job_owner_reference(self):
        owner_ref = k8sClient.create_owner_reference(
            api_version="elastic.iml.github.io/v1alpha1",
            kind="ElasticJob",
            name=self._job["metadata"]["name"],
            uid=self._job["metadata"]["uid"],
        )
        return owner_ref


def new_tf_config(
    pod_stats: Dict[str, int],
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
    if NodeType.WORKER in pod_stats:
        worker_num = pod_stats[NodeType.WORKER]
        if type_key == NodeType.WORKER and index_key >= worker_num:
            worker_num = index_key + 1
        workers = []
        for worker_id in range(worker_num):
            workers.append(new_service_fn(NodeType.WORKER, worker_id))
        if len(workers) > 0:
            cluster_dict[NodeType.WORKER] = workers
    if NodeType.EVALUATOR in pod_stats:
        evaluator_num = pod_stats[NodeType.EVALUATOR]
        evaluators = []
        for worker_id in range(evaluator_num):
            evaluators.append(new_service_fn(NodeType.EVALUATOR, worker_id))
        if len(evaluators) > 0:
            cluster_dict[NodeType.EVALUATOR] = evaluators
    if NodeType.CHIEF in pod_stats:
        chief_num = pod_stats[NodeType.CHIEF]
        chiefs = []
        for worker_id in range(chief_num):
            chiefs.append(new_service_fn(NodeType.CHIEF, worker_id))
        if len(chiefs) > 0:
            cluster_dict[NodeType.CHIEF] = chiefs

    task_dict = {}
    task_dict["type"] = type_key
    task_dict["index"] = index_key
    return {"cluster": cluster_dict, "task": task_dict}
