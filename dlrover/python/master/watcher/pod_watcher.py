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
    ElasticJobLabel,
    ExitCode,
    NodeExitReason,
    NodeType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
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
        if terminated.reason != "OOMKilled":
            if (
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
                return NodeExitReason.UNKNOWN_ERROR
        else:
            return NodeExitReason.OOM


def _convert_pod_event_to_node_event(event):
    evt_obj = event.get("object")
    evt_type = event.get("type")
    if not evt_obj or not evt_type:
        logger.error("Event doesn't have object or type: %s" % event)
        return None

    if evt_obj.kind != "Pod":
        # We only care about pod related events
        return None

    pod_type = evt_obj.metadata.labels[ElasticJobLabel.REPLICA_TYPE_KEY]
    pod_name = evt_obj.metadata.name
    rank = int(evt_obj.metadata.labels[ElasticJobLabel.RANK_INDEX_KEY])

    pod_id = int(evt_obj.metadata.labels[ElasticJobLabel.REPLICA_INDEX_KEY])

    resource = _parse_container_resource(evt_obj.spec.containers[0])
    node = Node(
        node_type=pod_type,
        node_id=pod_id,
        name=pod_name,
        rank_index=rank,
        status=evt_obj.status.phase,
        start_time=_get_start_timestamp(evt_obj.status),
        config_resource=resource,
    )
    node.set_exit_reason(_get_pod_exit_reason(evt_obj))
    node_event = NodeEvent(event_type=evt_type, node=node)
    return node_event


def _parse_container_resource(container):
    cpu = container.resources.requests["cpu"]
    memory = NodeResource.convert_memory_to_mb(
        container.resources.requests["memory"]
    )
    return NodeResource(cpu, memory)


class PodWatcher(NodeWatcher):
    """PodWatcher monitors all Pods of a k8s Job."""

    def __init__(self, job_name, namespace):
        self._job_name = job_name
        self._namespace = namespace
        self._k8s_client = k8sClient.singleton_instance(job_name, namespace)
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
        if not pod_list or not pod_list.items:
            return nodes

        for pod in pod_list.items:
            pod_type = pod.metadata.labels[ElasticJobLabel.REPLICA_TYPE_KEY]
            pod_id = int(
                pod.metadata.labels[ElasticJobLabel.REPLICA_INDEX_KEY]
            )
            if pod_type == NodeType.DLROVER_MASTER:
                continue
            task_id = int(pod.metadata.labels[ElasticJobLabel.RANK_INDEX_KEY])
            resource = _parse_container_resource(pod.spec.containers[0])
            node = Node(
                node_type=pod_type,
                node_id=pod_id,
                name=pod.metadata.name,
                rank_index=task_id,
                status=pod.status.phase,
                start_time=_get_start_timestamp(pod.status),
                config_resource=resource,
            )
            node.set_exit_reason(_get_pod_exit_reason(pod))
            nodes.append(node)
        return nodes
