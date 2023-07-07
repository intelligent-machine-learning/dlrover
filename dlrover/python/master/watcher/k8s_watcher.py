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

from typing import List

from kubernetes import watch

from dlrover.python.common.constants import (
    ElasticJobApi,
    ElasticJobLabel,
    ExitCode,
    NodeExitReason,
    NodeType,
    ScalePlanLabel,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.watcher.base_watcher import NodeEvent, NodeWatcher
from dlrover.python.scheduler.kubernetes import k8sClient


def _get_start_timestamp(pod_status_obj):
    """Get the start timestamp of a Pod"""
    if (
        pod_status_obj.container_statuses
        and pod_status_obj.container_statuses[0].state
        and pod_status_obj.container_statuses[0].state.running
    ):
        return pod_status_obj.container_statuses[0].state.running.started_at
    return None


def _get_pod_exit_reason(pod):
    """Get the exit reason of a Pod"""
    if (
        pod.status.container_statuses
        and pod.status.container_statuses[0].state.terminated
    ):
        terminated = pod.status.container_statuses[0].state.terminated
        if (
            terminated.reason == "OOMKilled"
            or terminated.exit_code == ExitCode.OOM_CODE
        ):
            return NodeExitReason.OOM
        elif (
            terminated.exit_code == ExitCode.KILLED_CODE
            or terminated.exit_code == ExitCode.TERMED_CODE
        ):
            return NodeExitReason.KILLED
        elif terminated.exit_code in (
            ExitCode.FATAL_ERROR_CODE,
            ExitCode.CORE_DUMP_ERROR_CODE,
        ):
            return NodeExitReason.FATAL_ERROR
        else:
            if terminated.exit_code in (
                ExitCode.GPU_DRIVER_ERROR,
                ExitCode.GPU_POD_RESIDUE,
            ):
                logger.info(
                    "Possible error found in GPU. Kill this node and launch a"
                    " new one."
                )
            return NodeExitReason.UNKNOWN_ERROR


def _convert_pod_event_to_node_event(event):
    evt_obj = event.get("object")
    evt_type = event.get("type")
    if not evt_obj or not evt_type:
        logger.error("Event doesn't have object or type: %s" % event)
        return None

    if evt_obj.kind != "Pod":
        # We only care about pod related events
        return None

    # Skip events of dlrover mater Pod
    pod_type = evt_obj.metadata.labels[ElasticJobLabel.REPLICA_TYPE_KEY]
    if pod_type == NodeType.DLROVER_MASTER:
        return None

    rank = int(evt_obj.metadata.labels[ElasticJobLabel.RANK_INDEX_KEY])
    pod_id = int(evt_obj.metadata.labels[ElasticJobLabel.REPLICA_INDEX_KEY])
    pod_name = evt_obj.metadata.name
    node_name = evt_obj.spec.node_name

    resource = _parse_container_resource(evt_obj.spec.containers[0])
    node = Node(
        node_type=pod_type,
        node_id=pod_id,
        name=pod_name,
        rank_index=rank,
        status=evt_obj.status.phase,
        start_time=_get_start_timestamp(evt_obj.status),
        config_resource=resource,
        node_name=node_name,
    )
    node.set_exit_reason(_get_pod_exit_reason(evt_obj))
    node_event = NodeEvent(event_type=evt_type, node=node)
    return node_event


def _parse_container_resource(container):
    cpu = NodeResource.convert_cpu_to_decimal(
        container.resources.requests["cpu"]
    )
    memory = NodeResource.convert_memory_to_mb(
        container.resources.requests["memory"]
    )
    return NodeResource(cpu, memory)


class PodWatcher(NodeWatcher):
    """PodWatcher monitors all Pods of a k8s Job."""

    def __init__(self, job_name, namespace):
        self._job_name = job_name
        self._namespace = namespace
        self._k8s_client = k8sClient.singleton_instance(namespace)
        self._job_selector = ElasticJobLabel.JOB_KEY + "=" + self._job_name

    def watch(self):
        resource_version = None
        pod_list = self._k8s_client.list_namespaced_pod(self._job_selector)
        if pod_list:
            resource_version = pod_list.metadata.resource_version
        try:
            stream = watch.Watch().stream(
                self._k8s_client.client.list_namespaced_pod,
                self._namespace,
                label_selector=self._job_selector,
                resource_version=resource_version,
                timeout_seconds=60,
            )
            for event in stream:
                node_event = _convert_pod_event_to_node_event(event)
                if not node_event:
                    continue
                yield node_event
        except Exception as e:
            raise e

    def list(self) -> List[Node]:
        nodes: List[Node] = []
        pod_list = self._k8s_client.list_namespaced_pod(self._job_selector)
        if not pod_list:
            return nodes
        if not pod_list.items:
            return nodes

        replica_type_key = ElasticJobLabel.REPLICA_TYPE_KEY
        replica_index_key = ElasticJobLabel.REPLICA_INDEX_KEY
        rank_index_key = ElasticJobLabel.RANK_INDEX_KEY

        for pod in pod_list.items:
            pod_type = pod.metadata.labels[replica_type_key]
            if pod_type == NodeType.DLROVER_MASTER:
                continue
            pod_id = int(pod.metadata.labels[replica_index_key])
            task_id = int(pod.metadata.labels[rank_index_key])
            resource = _parse_container_resource(pod.spec.containers[0])
            status = pod.status.phase
            start_time = _get_start_timestamp(pod.status)
            node = Node(
                node_type=pod_type,
                node_id=pod_id,
                name=pod.metadata.name,
                rank_index=task_id,
                status=status,
                start_time=start_time,
                config_resource=resource,
            )
            node.set_exit_reason(_get_pod_exit_reason(pod))
            nodes.append(node)
        return nodes


class K8sScalePlanWatcher:
    """ScalePlanWatcher monitors the manual Scaler CRDs on the cluster.
    It generates a ResourcePlan by a Scaler CRD and notidy the
    JobManager to adjust job resource by the ResourcePlan.
    """

    def __init__(self, job_name, namespace, job_uid):
        self._namespace = namespace
        self._job_name = job_name
        self._job_uid = job_uid
        self._k8s_client = k8sClient.singleton_instance(namespace)
        self._used_scaleplans = []
        job_label = "{}={}".format(ElasticJobLabel.JOB_KEY, self._job_name)
        type_label = "{}={}".format(
            ScalePlanLabel.SCALE_TYPE_KEY, ScalePlanLabel.MANUAL_SCALE
        )
        self._job_selector = job_label + "," + type_label

    def watch(self):
        resource_version = None
        try:
            stream = watch.Watch().stream(
                self._k8s_client.api_instance.list_namespaced_custom_object,
                namespace=self._namespace,
                group=ElasticJobApi.GROUP,
                version=ElasticJobApi.VERION,
                plural=ElasticJobApi.SCALEPLAN_PLURAL,
                label_selector=self._job_selector,
                resource_version=resource_version,
                timeout_seconds=60,
            )
            for event in stream:
                scaler_crd = event.get("object", None)
                evt_type = event.get("type")
                uid = scaler_crd["metadata"]["uid"]
                if (
                    evt_type != "ADDED"
                    or not scaler_crd
                    or scaler_crd["kind"] != "ScalePlan"
                    or uid in self._used_scaleplans
                ):
                    logger.info("Ignore an event")
                    continue
                self._used_scaleplans.append(uid)
                resource_plan = self._get_resoruce_plan_from_event(scaler_crd)
                self._set_owner_to_scaleplan(scaler_crd)
                yield resource_plan
        except Exception as e:
            raise e

    def _get_resoruce_plan_from_event(self, scaler_crd) -> ResourcePlan:
        resource_plan = ResourcePlan()
        for replica, spec in (
            scaler_crd["spec"].get("replicaResourceSpecs", {}).items()
        ):
            cpu = NodeResource.convert_cpu_to_decimal(
                spec.get("resource", {}).get("cpu", "0")
            )
            memory = NodeResource.convert_memory_to_mb(
                spec.get("resource", {}).get("memory", "0Mi")
            )
            resource_plan.node_group_resources[replica] = NodeGroupResource(
                spec["replicas"], NodeResource(cpu, memory)
            )

        for pod in scaler_crd["spec"].get("migratePods", []):
            cpu = NodeResource.convert_cpu_to_decimal(
                pod["resource"].get("cpu", "0"),
            )
            memory = NodeResource.convert_memory_to_mb(
                pod["resource"].get("memory", "0Mi"),
            )
            resource_plan.node_resources[pod["name"]] = NodeResource(
                cpu, memory
            )
        logger.info("Get a manual resource plan %s", resource_plan.toJSON())
        return resource_plan

    def _set_owner_to_scaleplan(self, scale_crd):
        api_version = ElasticJobApi.GROUP + "/" + ElasticJobApi.VERION
        ref_dict = {}
        ref_dict["apiVersion"] = api_version
        ref_dict["blockOwnerDeletion"] = True
        ref_dict["kind"] = ElasticJobApi.ELASTICJOB_KIND
        ref_dict["name"] = self._job_name
        ref_dict["uid"] = self._job_uid
        scale_crd["metadata"]["ownerReferences"] = [ref_dict]
        self._k8s_client.patch_custom_resource(
            group=ElasticJobApi.GROUP,
            version=ElasticJobApi.VERION,
            plural=ElasticJobApi.SCALEPLAN_PLURAL,
            name=scale_crd["metadata"]["name"],
            body=scale_crd,
        )
