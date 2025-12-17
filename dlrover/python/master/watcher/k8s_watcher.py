# Copyright 2025 The DLRover Authors. All rights reserved.
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
import threading
from datetime import datetime
from time import sleep
from typing import List

from kubernetes import client, watch

from dlrover.python.common.constants import (
    ElasticJobApi,
    ElasticJobLabel,
    ExitCode,
    JobConstant,
    JobStage,
    NodeEventType,
    NodeExitReason,
    NodeStatus,
    NodeType,
    ScalePlanLabel,
    SchedulingLabel,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import (
    Node,
    NodeEvent,
    NodeGroupResource,
    NodeResource,
)
from dlrover.python.master.node.job_context import JobContext, get_job_context
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.watcher.base_watcher import NodeWatcher
from dlrover.python.scheduler.kubernetes import (
    convert_cpu_to_decimal,
    convert_memory_to_mb,
    k8sClient,
)
from dlrover.python.common.global_context import (
    Context,
)

job_ctx = get_job_context()
_dlrover_context = Context.singleton_instance()


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
        exit_code = terminated.exit_code
        if terminated.reason == "OOMKilled" or exit_code == ExitCode.OOM_CODE:
            return NodeExitReason.OOM
        elif exit_code in [ExitCode.KILLED_CODE, ExitCode.TERMED_CODE]:
            # get extension info from pod meta
            extension_reason = _dlrover_context.get_k8s_util().resolve_extension_exit_reason_from_meta(
                pod.metadata
            )
            if extension_reason:
                logger.info(f"Got extension exit reason: {extension_reason} for exit code: {exit_code}")
                return extension_reason

            return NodeExitReason.KILLED
        elif exit_code in (
            ExitCode.FATAL_ERROR_CODE,
            ExitCode.CORE_DUMP_ERROR_CODE,
        ):
            return NodeExitReason.FATAL_ERROR
        elif exit_code in (
            ExitCode.GPU_DRIVER_ERROR,
            ExitCode.GPU_POD_RESIDUE,
            ExitCode.GPU_INFOROM_CORRUPTED,
        ):
            logger.info(
                "Possible error found in GPU. Will relaunch this node."
            )
            return NodeExitReason.HARDWARE_ERROR
        elif exit_code == 0:
            return NodeExitReason.Succeeded
        else:
            return NodeExitReason.UNKNOWN_ERROR

    # get extension info from pod meta if no exit code
    extension_reason = _dlrover_context.get_k8s_util().resolve_extension_exit_reason_from_meta(
        pod.metadata
    )
    if extension_reason:
        logger.info(f"Got extension exit reason: {extension_reason}")
        return extension_reason

    return ""


def _convert_pod_yaml_to_node(pod):
    replica_type_key = ElasticJobLabel.REPLICA_TYPE_KEY
    replica_index_key = ElasticJobLabel.REPLICA_INDEX_KEY
    rank_index_key = ElasticJobLabel.RANK_INDEX_KEY
    relaunch_count_key = ElasticJobLabel.RELAUNCH_COUNT

    metadata: client.V1ObjectMeta = pod.metadata
    pod_name = metadata.name
    pod_type = metadata.labels[replica_type_key]
    labels = metadata.labels
    node_group = None
    node_group_size = None
    node_group_id = None

    if pod_type == NodeType.DLROVER_MASTER:
        return None
    elif pod_type == NodeType.WORKER:
        try:
            if SchedulingLabel.NODE_GROUP in labels:
                node_group = int(labels[SchedulingLabel.NODE_GROUP])
                if SchedulingLabel.NODE_GROUP_SIZE in labels:
                    node_group_size = int(
                        labels[SchedulingLabel.NODE_GROUP_SIZE]
                    )
                if SchedulingLabel.NODE_GROUP_ID in labels:
                    node_group_id = labels[SchedulingLabel.NODE_GROUP_ID]
        except Exception as e:
            logger.error(
                f"Unexpected exception {e} on parsing {labels} "
                f"with {pod_name} {pod_type}"
            )
            node_group = None
            node_group_size = None
            node_group_id = None

    pod_id = int(metadata.labels[replica_index_key])
    rank_id = int(metadata.labels[rank_index_key])
    relaunch_count = int(metadata.labels[relaunch_count_key])
    resource = NodeResource(0, 0)
    # this is a temporary workaround to retrieve the 'main' container's resource
    for container in pod.spec.containers:
        res = _parse_container_resource(container)
        if res.cpu > 0 and res.memory > 0:
            resource = res
            break
    host_name = pod.spec.node_name
    host_ip = pod.status.host_ip

    # if pod has 'deletion_timestamp', set as deleted status directly
    # because the deletion has low probability of failure will affect
    # node status judgement
    if metadata.deletion_timestamp:
        status = NodeStatus.DELETED
    else:
        status = pod.status.phase

    start_time = _get_start_timestamp(pod.status)
    restart_training = _verify_restarting_training(pod)

    node = Node(
        node_type=pod_type,
        node_id=pod_id,
        name=pod_name,
        rank_index=rank_id,
        status=status,
        start_time=start_time,
        config_resource=resource,
        host_name=host_name,
        host_ip=host_ip,
        restart_training=restart_training,
        relaunch_count=relaunch_count,
        node_group=node_group,
        node_group_size=node_group_size,
        node_group_id=node_group_id,
        max_relaunch_count=_dlrover_context.max_relaunch_count,
    )

    logger.debug(
        f"convert yaml to node: {node} "
        f"type {pod_type}, id {pod_id}, name {pod_name}, "
        f"rank {rank_id}, status {status}, "
        f"group {node_group}, group_size {node_group_size}, "
        f"group_id {node_group_id}, with meta {metadata.labels}"
    )

    node.create_time = metadata.creation_timestamp
    if NodeStatus.is_terminal_status(status):
        node.set_exit_reason(_get_pod_exit_reason(pod))

    return node


def _convert_pod_event_to_node_event(event):
    pod = event.get("object")
    evt_type = event.get("type")
    if not pod or not evt_type:
        logger.error("Event doesn't have object or type: %s" % event)
        return None

    if pod.kind != "Pod":
        # We only care about pod related events
        return None

    node = _convert_pod_yaml_to_node(pod)
    if node is None:
        return None

    logger.debug(
        f"Got monitor event for pod: {node.name}, "
        f"node: {node.host_name}, ip: {node.host_ip}, "
        f"status: {node.status}."
    )
    if node.restart_training:
        logger.info(f"{node.name} need to restart.")

    node_event = NodeEvent(event_type=evt_type, node=node)
    return node_event


def _parse_container_resource(container):
    cpu = convert_cpu_to_decimal(container.resources.requests["cpu"])
    memory = convert_memory_to_mb(container.resources.requests["memory"])
    return NodeResource(cpu, memory)


def _verify_restarting_training(pod):
    if not pod.metadata.annotations:
        return False

    action_str = pod.metadata.annotations.get(
        _dlrover_context.get_k8s_util().get_annotation_scheduled_action()
    )
    if not action_str:
        return False
    action_config = json.loads(action_str)
    action = action_config.get("scheduledAction", "")
    if action == "RestartTrain_Observe":
        return True
    return False


def _get_pod_unique_labels(job_name, pod_type, rank_index):
    return {
        ElasticJobLabel.JOB_KEY: job_name,
        ElasticJobLabel.REPLICA_TYPE_KEY: pod_type,
        ElasticJobLabel.RANK_INDEX_KEY: rank_index,
    }


class PodWatcher(NodeWatcher):
    """PodWatcher monitors all Pods of a k8s Job."""

    def __init__(self, job_name, namespace):
        super().__init__(job_name)
        self._job_name = job_name
        self._namespace = namespace
        self._k8s_client = k8sClient.singleton_instance(namespace)
        self._job_selector = ElasticJobLabel.JOB_KEY + "=" + self._job_name
        logger.info(
            f"Initialize PodWatcher with "
            f"namespace: {self._namespace}, "
            f"job-selector: {self._job_selector}"
        )

    def watch(self):
        resource_version = None
        pod_list = self._k8s_client.list_namespaced_pod(self._job_selector)
        if pod_list:
            resource_version = pod_list.metadata.resource_version

        w = watch.Watch()
        try:
            stream = w.stream(
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
        finally:
            w.stop()

    def list(self) -> List[Node]:
        nodes: List[Node] = []
        pod_list = self._k8s_client.list_namespaced_pod(self._job_selector)
        if not pod_list:
            return nodes
        if not pod_list.items:
            return nodes

        for pod in pod_list.items:
            node = _convert_pod_yaml_to_node(pod)
            if node is None:
                continue
            nodes.append(node)

            # delete pod if pod already succeeded(no need for failed pod,
            # cuz the deletion will be done in relaunch operation)
            if pod.status.phase == NodeStatus.SUCCEEDED:
                logger.info(f"Delete succeeded pod: {node.name}")
                self._k8s_client.delete_pod(node.name)
            else:
                target_node = job_ctx.job_node(node.type, node.id)
                now = int(datetime.now().timestamp())
                if target_node:
                    status, ts = target_node.reported_status
                    if (
                        status == NodeEventType.SUCCEEDED_EXITED
                        and now - ts
                        > JobConstant.SUCCEEDED_POD_TERMINATING_TIMEOUT
                    ):
                        logger.info(
                            f"Delete target pod {node.name} due to "
                            f"report status {status} from {ts} to {now} "
                            f"has exceeded 600s"
                        )
                        self._k8s_client.delete_pod(node.name)

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
            cpu = convert_cpu_to_decimal(
                spec.get("resource", {}).get("cpu", "0")
            )
            memory = convert_memory_to_mb(
                spec.get("resource", {}).get("memory", "0Mi")
            )
            resource_plan.node_group_resources[replica] = NodeGroupResource(
                spec["replicas"], NodeResource(cpu, memory)
            )

        for pod in scaler_crd["spec"].get("migratePods", []):
            cpu = convert_cpu_to_decimal(
                pod["resource"].get("cpu", "0"),
            )
            memory = convert_memory_to_mb(
                pod["resource"].get("memory", "0Mi"),
            )
            resource_plan.node_resources[pod["name"]] = NodeResource(
                cpu, memory
            )
        logger.info("Get a manual resource plan %s", resource_plan.to_json())
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


class K8sElasticJobWatcher(object):
    """K8sElasticJobWatcher monitors the Elasticjob CR on the cluster.
    It nodify the JobContext to update the job status.
    """

    def __init__(self, args):
        self._job_name = args.job_name
        self._namespace = args.namespace
        self._job_uid = args.job_uuid
        self._enable_suspended = args.enable_suspended
        self._k8s_client = k8sClient.singleton_instance(args.namespace)
        self._job_context = JobContext.singleton_instance()
        self._job_pre_status = JobStage.JOB_INIT

    def watch(self):
        w = watch.Watch()
        api_instance = self._k8s_client.api_instance
        while True:
            try:
                for event in w.stream(
                    api_instance.list_namespaced_custom_object,
                    namespace=self._namespace,
                    group=ElasticJobApi.GROUP,
                    version=ElasticJobApi.VERION,
                    plural=ElasticJobApi.ELASTICJOB_PLURAL,
                    timeout_seconds=60,
                ):
                    logger.debug(f"get elasticjob event, {event}")
                    elasticjob_cr = event.get("object", None)
                    evt_type = event.get("type")
                    if (
                        evt_type == "MODIFIED" or evt_type == "ADDED"
                    ) and elasticjob_cr["metadata"].get(
                        "name", ""
                    ) == self._job_name:
                        logger.info(f"get elasticjob {evt_type} event")

                        enable_suspended = elasticjob_cr["spec"].get(
                            "suspend", False
                        )
                        if (
                            enable_suspended
                            and not self._job_context.is_suspended()
                        ):
                            logger.info("try to request suspend")
                            self._job_context.request_suspend()
                        if (
                            not enable_suspended
                            and self._job_context.is_suspended()
                        ):
                            logger.info("try to request unsuspend")
                            self._job_context.request_unsuspend()

                sleep(5)
            except Exception as e:
                logger.warning(e)
                sleep(5)

    def start(self):
        if self._enable_suspended:
            self._job_context.request_suspend()

        threading.Thread(
            target=self.watch, name="job-watcher", daemon=True
        ).start()
