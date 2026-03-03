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
import os
import socket
import telnetlib
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Dict, List, Optional

from kubernetes import client
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector

from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    SchedulingLabel,
    EventReportConstants,
    NodeEnv,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.scheduler.kubernetes import (
    NODE_SERVICE_PORTS,
    convert_cpu_to_decimal,
    convert_memory_to_mb,
    get_main_container,
    get_pod_name,
    k8sClient,
    k8sServiceFactory,
    set_container_resource,
)

_dlrover_context = Context.singleton_instance()
_job_context = get_job_context()


class FakeKubeResponse:
    def __init__(self, obj):
        self.data = json.dumps(obj)


def append_pod_ip_to_env(env):
    pod_ip_var = V1EnvVar(
        name="POD_IP",
        value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.podIP")
        ),
    )
    node_ip_var = V1EnvVar(
        name="NODE_IP",
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
        self._svc_factory = k8sServiceFactory(namespace, job_name)
        self._namespace = namespace
        self._replica_template: Dict[str, client.V1Pod] = {}
        self._create_node_queue: Deque[Node] = deque()
        self._create_node_futures: List[Future] = []
        self._scaling_lock = threading.Lock()
        self._plan = ScalePlan()
        self._ps_addrs: List[str] = []
        self._pod_stats: Dict[str, int] = {}
        self._alive_pod_stats: Dict[str, int] = {}
        self._var_lock = threading.Lock()
        self._job_uid = ""
        self.api_client = client.ApiClient()
        self._master_addr = ""
        self._master_service_type = _dlrover_context.master_service_type
        self._event_reporter = get_event_reporter()
        self._started = False
        self._job_context = get_job_context()

        # for concurrency scaling
        self._pending_plan: Optional[ScalePlan] = None
        self._plan_merge_lock = threading.Lock()
        self._current_scaling: Optional[Future] = None
        self._scaling_executor = ThreadPoolExecutor(max_workers=1)

    def start(self):
        self._job = self._retry_to_get_job()
        if not self._job:
            raise ValueError(f"Cannot get the training job {self._job_name}.")
        self._init_pod_config_by_job()
        self._master_addr = self._get_master_addr()
        self._started = True
        threading.Thread(
            target=self._periodic_create_pod, name="pod-creater", daemon=True
        ).start()

    def stop(self):
        self._started = False

    def _safe_get_pod_status(self, key, default):
        with self._var_lock:
            if key is None:
                # return whole dict
                return self._pod_stats
            # return specified value
            return self._pod_stats.get(key, default)

    def _safe_set_pod_status(self, key, value):
        with self._var_lock:
            logger.info(f"Set pod number {value} for {key}.")
            self._pod_stats[key] = value

    def _safe_get_alive_pod_status(self, key, default):
        with self._var_lock:
            if key is None:
                # return whole dict
                return self._alive_pod_stats
            # return specified value
            return self._alive_pod_stats.get(key, default)

    def _safe_set_alive_pod_status(self, key, value):
        with self._var_lock:
            logger.info(f"Set alive pod number {value} for {key}.")
            self._alive_pod_stats[key] = value

    def _get_master_addr(self):
        svc_name = f"elasticjob-{self._job_name}-dlrover-master"
        port = NODE_SERVICE_PORTS[NodeType.DLROVER_MASTER]
        master_addr = f"{svc_name}:{port}"
        target_port = _dlrover_context.master_port
        if not self._check_master_service_avaliable(svc_name, target_port):
            # On some clusters, the k8s service may not be available because of
            # incorrect DNS configurations. In such cases, it is necessary to
            # revert to use the Master's IP address for worker to connect with
            # the master. Note, that failover of master is not supported if
            # the service is not available.
            logger.info(
                f"The service {master_addr} is not available and "
                f"use the IP of master Pod."
            )
            master_ip = os.getenv("POD_IP", "")
            if not master_ip:
                raise ValueError("The master Pod must have the POD_IP env.")
            master_addr = f"{master_ip}:{_dlrover_context.master_port}"
        return master_addr

    def _init_pod_config_by_job(self):
        self._distribution_strategy = self._job["spec"].get(
            "distributionStrategy", None
        )
        self._job_uid = self._job["metadata"]["uid"]
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
        return None

    def scale(self, plan: ScalePlan, **kwargs):
        """
        Scale with automatic plan merging and execution(set with_merge=True).
        If a scaling task is running, merge the new plan into pending plan.
        When current task finishes, automatically execute the merged pending plan.

        Args:
            plan (ScalePlan): The scaling plan to execute
            with_merge (bool, optional): If true, enable automatic plan merging.
                Defaults to False.
        """

        with_merge = kwargs.pop("with_merge", False)

        if with_merge:
            with self._plan_merge_lock:
                if (
                    self._current_scaling is None
                    or self._current_scaling.done()
                ):
                    # no scaling running, execute immediately
                    self._current_scaling = self._scaling_executor.submit(
                        self._execute_scale_with_pending, plan
                    )
                else:
                    # scaling is running, merge into pending plan
                    if self._pending_plan is None:
                        self._pending_plan = copy.deepcopy(plan)
                    else:
                        self._pending_plan.merge(plan)
                    logger.info(
                        "Merged plan into pending, "
                        f"current launch nodes: {len(self._pending_plan.launch_nodes)}, "
                        f"remove nodes: {len(self._pending_plan.remove_nodes)}"
                        f"full plan: {self._pending_plan.to_json()}"
                    )
        else:
            self._scale(plan)

    def _execute_scale_with_pending(self, plan: ScalePlan):
        """
        Execute scale plan and handle pending plans after completion.
        This method ensures pending plans are executed in sequence.
        """

        try:
            # execute the current plan
            self._scale(plan)
        except Exception:
            logger.exception(f"Failed to execute scale plan: {plan.to_json()}")
        finally:
            # after completion, check and execute any pending plans
            with self._plan_merge_lock:
                if self._pending_plan is None or self._pending_plan.empty():
                    # no more pending plans, mark as completed
                    self._current_scaling = None
                    return

                # get the pending plan and clear it
                next_plan = copy.deepcopy(self._pending_plan)
                self._pending_plan = None

            logger.info(f"Executing pending plan: {next_plan.to_json()}")

            # execute the pending plan
            self._current_scaling = self._scaling_executor.submit(
                self._execute_scale_with_pending, next_plan
            )

    def _scale(self, plan: ScalePlan):
        if not self._elasticjob_exists():
            plan_json = plan.to_json()
            logger.info(
                f"Skip the scale plan {plan_json} because the job does not exist."
            )
            return

        while True:
            if not self._job_context.is_suspended():
                break
            logger.info("Waiting for elasticJob which is suspended")
            time.sleep(5)

        self._remove_nodes(plan)
        while self._started:
            if (
                len(self._create_node_queue) > 0
                and not _job_context.is_stopped()
            ):
                logger.info(
                    f"Wait nodes {self._create_node_queue} to completed."
                )
                time.sleep(15)
            else:
                if all(future.done() for future in self._create_node_futures):
                    # wait async pod creation completed
                    logger.debug("Async pod creation finished.")
                    time.sleep(1)
                    break
                else:
                    logger.debug("Waiting for async pod creation...")
                    time.sleep(1)
                    continue

        with self._scaling_lock:
            if plan.empty():
                return
            self._plan = plan
            job_pods = self._list_job_pods()
            logger.info("Scale the job by plan %s", plan.to_json())
            if plan.ps_addrs:
                self._ps_addrs = plan.ps_addrs
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
            self._update_job_pods(job_pods)

    def _remove_nodes(self, plan: ScalePlan):
        for node in plan.remove_nodes:
            removed = self._remove_not_create_pod(node.name)
            if not removed and node.name:
                self._k8s_client.delete_pod(node.name)

    def _update_job_pods(self, job_pods: Dict[str, List[Node]]):
        for node_type in [
            NodeType.CHIEF,
            NodeType.MASTER,
            NodeType.PS,
            NodeType.WORKER,
            NodeType.EVALUATOR,
        ]:
            cur_pods = job_pods.get(
                node_type, []
            ) + self._get_type_pod_in_queue(node_type)
            self._safe_set_pod_status(node_type, len(cur_pods))
            self._safe_set_alive_pod_status(
                node_type,
                len(
                    [
                        cur_pod
                        for cur_pod in cur_pods
                        if cur_pod.status != NodeStatus.FAILED
                        and cur_pod.status != NodeStatus.DELETED
                    ]
                ),
            )

    def _get_type_pod_in_queue(self, node_type):
        pods = []
        for pod in self._create_node_queue:
            if pod.type == node_type:
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
        pod_list = self._wait_list_pods()
        job_pods: Dict[str, List[Node]] = {}
        if not pod_list:
            return job_pods
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
                max_relaunch_count=_dlrover_context.max_relaunch_count,
            )
            if node.type != NodeType.WORKER and node.status not in [
                NodeStatus.PENDING,
                NodeStatus.RUNNING,
            ]:
                continue
            job_pods[pod_type].append(node)
        return job_pods

    def _wait_list_pods(self, timeout=1800):
        job_selector = ElasticJobLabel.JOB_KEY + "=" + self._job_name
        start = time.time()
        while True:
            pod_list = self._k8s_client.list_namespaced_pod(job_selector)
            if pod_list:
                return pod_list
            if time.time() - start < timeout:
                time.sleep(60)
            else:
                raise TimeoutError(f"Timeout {timeout} to list Pods.")

    def _get_pod_resource(self, pod):
        resources = pod.spec.containers[0].resources
        cpu = convert_cpu_to_decimal(resources.requests.get("cpu", "0"))
        if "memory" in resources.requests:
            memory = convert_memory_to_mb(resources.requests["memory"])
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
        logger.info("Start the thread to create Pod.")
        with ThreadPoolExecutor(max_workers=4) as executor:
            while self._started:
                while self._create_node_queue:
                    self._create_node_futures.append(
                        executor.submit(
                            self._create_pod_from_queue,
                            self._create_node_queue.popleft(),
                        )
                    )
                time.sleep(3)

    def _create_pod_from_queue(self, node_from_queue=None):
        """
        Notice: we must ensure the sync operation of getting node happens
        before the async execution, so we set 'node_from_queue' in the params
        instead of pop the element in the current function to avoid invalid
        async execution calls.

        Args:
            node_from_queue (Node): List of Node instances.
        """

        if node_from_queue is None:
            return True

        succeed = False

        try:
            if self._check_cluster_ready_for_pod(node_from_queue):
                pod = self._create_pod(node_from_queue)
                succeed = self._k8s_client.create_pod(pod)
            if not succeed:
                self._create_node_queue.appendleft(node_from_queue)
            else:
                # create svs for succeed pod
                if not self._create_service_for_pod(node_from_queue):
                    self._create_node_queue.appendleft(node_from_queue)
        except Exception as e:
            logger.error(
                f"Failed to create pod by unexpected error: {e}", exc_info=True
            )
            succeed = False

        return succeed

    def _check_cluster_ready_for_pod(self, node: Node):
        """Check whether the resource of a cluster is enough to
        create a node"""
        return True

    def _create_pod(self, node: Node):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        node.update_priority(self._safe_get_pod_status(node.type, 0))
        pod_name = get_pod_name(self._job_name, node.type, node.id)
        logger.info(
            "Create Pod %s with resource %s",
            pod_name,
            node.config_resource.to_resource_dict(),
        )
        env: List[V1EnvVar] = []
        env = append_pod_ip_to_env(env)

        env.append(V1EnvVar(name=NodeEnv.JOB_NAME, value=self._job_name))
        env.append(V1EnvVar(name=NodeEnv.JOB_UID, value=self._job_uid))

        # History background1: https://chromium.googlesource.com/external/
        # github.com/grpc/grpc/+/refs/tags/v1.19.0-pre1/doc/fork_support.md
        #
        # History background2: https://github.com/grpc/grpc/issues/18075
        # resolved by: https://github.com/grpc/grpc/pull/32935
        env.append(V1EnvVar(name=NodeEnv.GRPC_ENABLE_FORK, value="false"))

        worker_num = self._config_worker_num
        if worker_num == 0:
            worker_num = self._safe_get_pod_status(node.type, 0)

        env.append(V1EnvVar(name=NodeEnv.NODE_TYPE, value=node.type))
        env.append(V1EnvVar(name=NodeEnv.NODE_ID, value=str(node.id)))
        env.append(V1EnvVar(name=NodeEnv.NODE_NUM, value=str(worker_num)))
        env.append(
            V1EnvVar(name=NodeEnv.NODE_RANK, value=str(node.rank_index))
        )

        # The two env vars is compatible with kubeflow/PytorchJob because
        # users may use the scripts of kubeflow/PytorchJob in the ElasticJob.
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            env.append(
                V1EnvVar(name=NodeEnv.WORLD_SIZE, value=str(worker_num))
            )
            env.append(V1EnvVar(name=NodeEnv.RANK, value=str(node.rank_index)))

        env.append(
            V1EnvVar(name=NodeEnv.DLROVER_MASTER_ADDR, value=self._master_addr)
        )
        env.append(
            V1EnvVar(
                name=NodeEnv.DLROVER_MASTER_SERVICE_TYPE,
                value=self._master_service_type,
            )
        )
        env.append(
            V1EnvVar(
                name=NodeEnv.DLROVER_TRAINING_ELASTIC_MODE,
                value=_dlrover_context.training_elastic_mode,
            )
        )

        env.append(
            V1EnvVar(
                name=NodeEnv.POD_NAME,
                value_from=V1EnvVarSource(
                    field_ref=V1ObjectFieldSelector(field_path="metadata.name")
                ),
            )
        )
        env.append(
            V1EnvVar(
                name=NodeEnv.DLROVER_EXTENSION_DYNAMIC_FAILOVER,
                value=_dlrover_context.dynamic_failover_extension,
            )
        )

        # Deprecated env vars
        env.append(V1EnvVar(name=NodeEnv.WORKER_TYPE, value=node.type))
        env.append(V1EnvVar(name=NodeEnv.WORKER_ID, value=str(node.id)))
        env.append(V1EnvVar(name=NodeEnv.WORKER_NUM, value=str(worker_num)))
        env.append(
            V1EnvVar(name=NodeEnv.WORKER_RANK, value=str(node.rank_index))
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
        labels = {
            "app": ElasticJobLabel.APP_NAME,
            ElasticJobLabel.JOB_KEY: self._job_name,
        }
        pod = self._create_pod_obj(
            name=pod_name,
            pod_template=pod_template,
            resource_requests=node.config_resource,
            resource_limits=node.config_resource,
            priority=node.config_resource.priority,
            env=env,
            lifecycle=None,
            labels=labels,
        )
        pod_meta: client.V1ObjectMeta = pod.metadata

        logger.debug(
            f"Add pod {pod_name} info into meta: {node.type} "
            f"{node.id} {node.rank_index} {node.relaunch_count} "
            f"{node.group} {node.group_size} {node.group_id}"
        )
        # Add replica type and index
        pod_meta.labels[ElasticJobLabel.REPLICA_TYPE_KEY] = node.type
        pod_meta.labels[ElasticJobLabel.REPLICA_INDEX_KEY] = str(node.id)
        pod_meta.labels[ElasticJobLabel.RANK_INDEX_KEY] = str(node.rank_index)
        pod_meta.labels[ElasticJobLabel.RELAUNCH_COUNT] = str(
            node.relaunch_count
        )
        if node.group is not None:
            pod_meta.labels[SchedulingLabel.NODE_GROUP] = str(node.group)
        if node.group_size is not None:
            pod_meta.labels[SchedulingLabel.NODE_GROUP_SIZE] = str(
                node.group_size
            )
        if node.group_id is not None:
            pod_meta.labels[SchedulingLabel.NODE_GROUP_ID] = str(node.group_id)
        pod.spec.containers[0].env.append(
            V1EnvVar(name=NodeEnv.MONITOR_ENABLED, value="true")
        )
        self._patch_tf_config_into_env(pod, node)
        if self._event_reporter:
            self._event_reporter.report(
                EventReportConstants.TYPE_INFO,
                pod_name,
                EventReportConstants.ACTION_WORKER_CREATE,
                "",
                {},
            )

        return pod

    def _check_master_service_avaliable(self, host, port, timeout=15):
        """Verify that the master grpc servicer is available."""
        for i in range(timeout):
            try:
                telnetlib.Telnet(host=host, port=port, timeout=3)
                logger.info(f"Master service check pass with {host}:{port}")
                return True
            except socket.gaierror:
                logger.info(
                    f"Attempt {i}: Encountered gaierror while "
                    "performing master service check. "
                    "Service may still be unavailable."
                )
                time.sleep(1)
            except Exception as e:
                logger.info(
                    f"Attempt {i}: Encountered {str(e)} while "
                    "performing master service check. "
                    "Service may still be unavailable."
                )
                time.sleep(1)

        logger.warning(
            f"Master service check {host}:{port} failed after {timeout} retries."
        )
        return False

    def _patch_tf_config_into_env(self, pod, node: Node):
        if (
            self._distribution_strategy == DistributionStrategy.PS
            and self._ps_addrs
        ):
            tf_config = new_tf_config(
                self._safe_get_pod_status(None, None),
                self.get_node_service_addr,
                node.type,
                node.rank_index,
                self._ps_addrs,
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
        logger.info(f"create service for node {node}")
        service_ready = True
        if node.service_addr:
            service_name = node.service_addr.split(".")[0]
        else:
            service_name = get_pod_name(
                self._job_name, node.type, node.rank_index
            )
        selector = {
            ElasticJobLabel.JOB_KEY: self._job_name,
            ElasticJobLabel.REPLICA_TYPE_KEY: node.type,
            ElasticJobLabel.RANK_INDEX_KEY: str(node.rank_index),
        }
        succeed = self._svc_factory.create_service(
            service_name,
            port=NODE_SERVICE_PORTS[node.type],
            target_port=NODE_SERVICE_PORTS[node.type],
            selector=selector,
            owner_ref=self._create_job_owner_reference(),
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

    def _create_pod_obj(
        self,
        name,
        pod_template: client.V1Pod,
        resource_requests: NodeResource,
        resource_limits: NodeResource,
        lifecycle,
        env,
        priority,
        labels,
        termination_period=None,
    ):
        pod = copy.deepcopy(pod_template)
        main_container: Optional[client.V1Container] = get_main_container(pod)

        if main_container is None:
            raise ValueError("The Pod config must have a main container.")
        set_container_resource(
            main_container, resource_requests, resource_limits
        )

        if main_container.env is None:
            main_container.env = env
        else:
            main_container.env.extend(env)

        main_container.lifecycle = lifecycle
        pod.spec.priority_class_name = priority
        pod.spec.restart_policy = "Never"
        pod.spec.termination_grace_period_seconds = termination_period

        if not pod.metadata:
            pod.metadata = client.V1ObjectMeta(
                name=name,
                namespace=self._namespace,
                labels=labels,
            )
        pod.metadata.name = name
        pod.metadata.namespace = self._namespace
        if pod.metadata.labels:
            pod.metadata.labels.update(labels)
        else:
            pod.metadata.labels = labels
        pod.metadata.owner_references = [self._create_job_owner_reference()]
        return pod

    def _create_job_owner_reference(self):
        owner_ref = k8sClient.create_owner_reference(
            api_version="elastic.iml.github.io/v1alpha1",
            kind="ElasticJob",
            name=self._job["metadata"]["name"],
            uid=self._job["metadata"]["uid"],
        )
        return owner_ref

    def _elasticjob_exists(self):
        job = self._retry_to_get_job()
        return job is not None


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
