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

import threading
import time
from collections import Counter
from typing import Dict, List

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeEventType,
    NodeExitReason,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.master.node_manager.event_callback import (
    ClusterContext,
    NodeEventCallback,
)
from dlrover.python.master.node_manager.job_config import (
    JobResourceConfig,
    get_critical_worker_index,
    set_critical_node,
)
from dlrover.python.master.node_manager.status_flow import (
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node_watcher.base_watcher import Node, NodeEvent
from dlrover.python.master.node_watcher.pod_watcher import PodWatcher
from dlrover.python.scheduler.kubernetes import k8sClient

_MAX_POD_RELAUNCH_COUNT = 5


class NodeManager(object):
    def __init__(
        self,
        job_resource,
        job_name,
        namespace,
        relaunch_on_worker_failure=0,
        ps_is_critical=True,
        critical_worker_index={},
        wait_pending_relaunch=False,
        ps_relaunch_max_num=1,
        use_ddp=False,
        speed_monitor=None,
    ):
        self._job_resource = job_resource
        self._relaunch_on_worker_failure = min(
            relaunch_on_worker_failure, _MAX_POD_RELAUNCH_COUNT
        )
        self._wait_pending_relaunch = wait_pending_relaunch
        self._start_launch_waiting_workers_time = time.time()
        self._stop_launch_worker_for_ps = False
        self._ps_is_critical = ps_is_critical
        self._critical_worker_index = critical_worker_index
        self._ps_relaunch_max_num = min(
            ps_relaunch_max_num, _MAX_POD_RELAUNCH_COUNT
        )
        self._use_ddp = use_ddp
        self._node_event_callbacks: List[NodeEventCallback] = []
        self._chief_worker_started = False
        self._stop_monitor = False
        self._last_pod_stats = None
        self._speed_monitor = speed_monitor

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()

        # TODO @workingloong bstract the k8sClient.
        self._k8s_client = k8sClient(namespace, job_name)
        self._job_nodes: Dict[str, Dict[int, Node]] = {}
        self._node_watcher = PodWatcher(job_name, namespace)
        self._job_uuid = None

    def start(self):
        self._job_uuid = self._k8s_client.get_job_uuid()
        self._init_job_nodes()
        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()

    def get_job_uuid(self):
        return self._job_uuid

    def add_node_event_callback(self, pod_event_callback):
        self._node_event_callbacks.append(pod_event_callback)

    def _init_job_nodes(self):
        self._job_nodes = self._job_resource.init_job_node_meta(
            self._relaunch_on_worker_failure,
            self._k8s_client.get_service_address,
        )

        # worker and eval ids for pods that should be created
        # after all ps are running.
        self._workers_waiting_ps_running = []

        # ps pod ids that are deleted and waiting for relaunch
        self._deleted_ps_pod_ids = []

        self._relaunch_pod = True
        self._pending_relaunch_count = 0

        set_critical_node(
            self._job_nodes,
            self._ps_is_critical,
            self._critical_worker_index,
            self._ps_relaunch_max_num,
        )

    def _monitor_nodes(self):
        while True:
            nodes = self._node_watcher.list()
            self._process_list_nodes(nodes)
            try:
                if self._stop_monitor:
                    logger.info("Stop processing node events")
                    break
                for event in self._node_watcher.watch():
                    try:
                        self._process_event(event)
                    except Exception as e:
                        logger.warning(e)
            except Exception as e:
                logger.warning(e)
                time.sleep(30)

    def _process_list_nodes(self, nodes: List[Node]):
        """Callback with pod list by the list api of k8s."""
        exist_pods: Dict[str, List[int]] = {}
        for node_type in self._job_nodes.keys():
            exist_pods[node_type] = []
        for node in nodes:
            exist_pods[node.type].append(node.id)
            if node.status == NodeStatus.DELETED:
                type = NodeEventType.DELETED
            else:
                type = NodeEventType.MODIFIED
            # Mock event to avoid missing events
            event = NodeEvent(type, node)
            self._process_event(event)

        for node_type in self._job_nodes.keys():
            for node_id, node in self._job_nodes[node_type].items():
                if (
                    node.status != NodeStatus.INITIAL
                    and not node.is_released
                    and node_id not in exist_pods[node_type]
                ):
                    logger.info(
                        "Node %s %s is deleted without the event",
                        node_type,
                        node_id,
                    )
                    node.is_released = True

    def _process_event(self, event: NodeEvent):
        node_type = event.node.type
        node_id = event.node.id
        cur_node = self._job_nodes[node_type][node_id]
        cur_node.update_info(
            name=event.node.name,
            start_time=event.node.start_time,
            create_time=event.node.create_time,
        )

        # For the given node id, check whether it meets
        # the state change condition
        new_status = event.node.status
        with self._lock:
            old_status = cur_node.status
            status_change_flow: NodeStateFlow = get_node_state_flow(
                old_status, event.event_type, new_status
            )
            cur_node.update_status(new_status)
            # If there is no matched state change, return directly
            # If the pod has been succeed, return directly
            if (
                status_change_flow is None
                or status_change_flow.from_status == NodeStatus.SUCCEEDED
            ):
                return

            # Update the pod status for pod_info
            new_status = status_change_flow.to_status
            cur_node.set_exit_reason(event.node.exit_reason)
            self._process_node_events(status_change_flow, cur_node)

            should_relaunch = self._should_relaunch(
                cur_node, status_change_flow
            )
            if should_relaunch and self._wait_pending_relaunch:
                self._pending_relaunch_count += 1

        logger.info(
            "%s status change: %s to %s, by evt_type %s, phase %s",
            cur_node.name,
            old_status,
            new_status,
            event.event_type,
            new_status,
        )

        if should_relaunch:
            self._relaunch_typed_pod(cur_node)

    def _process_node_events(
        self, status_change_flow: NodeStateFlow, node: Node
    ):
        cluster_context = ClusterContext(node_manager=self)
        if status_change_flow.to_status == NodeStatus.RUNNING:
            [
                callback.on_node_started(node, cluster_context)
                for callback in self._node_event_callbacks
            ]
        elif status_change_flow.to_status == NodeStatus.SUCCEEDED:
            [
                callback.on_node_succeeded(node, cluster_context)
                for callback in self._node_event_callbacks
            ]
        elif status_change_flow.to_status == NodeStatus.FAILED:
            [
                callback.on_node_failed(node, cluster_context)
                for callback in self._node_event_callbacks
            ]
        elif (
            status_change_flow.from_status != NodeStatus.FAILED
            and status_change_flow.from_status != NodeStatus.SUCCEEDED
            and status_change_flow.to_status == NodeStatus.DELETED
        ):
            [
                callback.on_node_deleted(node, cluster_context)
                for callback in self._node_event_callbacks
            ]

    def _should_relaunch(self, node: Node, status_change_flow: NodeStateFlow):
        should_relaunch = (
            status_change_flow.should_relaunch
            and self._relaunch_pod
            and node.relaunchable
        )
        if should_relaunch:
            if node.exit_reason == NodeExitReason.FATAL_ERROR:
                should_relaunch = False
            elif node.exit_reason == NodeExitReason.OOM:
                mem = node.used_resource.memory
                if mem > NodeResourceLimit.MAX_MEMORY:
                    should_relaunch = False
                    logger.warning(
                        "The memory of worker %s is beyond the limit %s MB.",
                        mem,
                        NodeResourceLimit.MAX_MEMORY,
                    )
                elif node.relaunch_count >= node.max_relaunch_count:
                    should_relaunch = False
                    logger.warning(
                        "The relaunched count %s is beyond the maximum %s.",
                        node.relaunch_count,
                        node.max_relaunch_count,
                    )
                else:
                    node.is_recovered_oom = True
            elif node.exit_reason != NodeExitReason.KILLED:
                if node.relaunch_count > node.max_relaunch_count:
                    logger.warning(
                        "The relaunch count for Error has been exhausted."
                    )
                    should_relaunch = False
        if should_relaunch:
            node.inc_relaunch_count()

        return should_relaunch

    def _relaunch_typed_pod(self, node: Node):
        logger.info("Relaunch the pod: {}".format(node.name))

    def all_workers_exited(self):
        counter = self._get_worker_status_counter()

        # At start, there may be no launched worker.
        if len(counter) == 1 and NodeStatus.INITIAL in counter:
            return False

        all_exited = True
        with self._lock:
            all_workers = (
                list(self._job_nodes[NodeType.WORKER].values())
                + list(self._job_nodes[NodeType.EVALUATOR].values())
                + list(self._job_nodes[NodeType.TF_MASTER].values())
            )
            for worker in all_workers:
                if not worker.is_released and (
                    worker.status
                    in [
                        NodeStatus.RUNNING,
                        NodeStatus.PENDING,
                        NodeStatus.INITIAL,
                    ]
                ):
                    all_exited = False
                    break
        return all_exited

    def all_workers_failed(self):
        counter = self._get_worker_status_counter()
        if len(counter) == 1 and NodeStatus.INITIAL in counter:
            return False

        all_failed = all(
            [
                status
                in [NodeStatus.FAILED, NodeStatus.DELETED, NodeStatus.INITIAL]
                for status in counter
            ]
        )
        return all_failed

    def all_workers_deleted(self):
        counter = self._get_worker_status_counter()
        all_deleted = all([status == NodeStatus.DELETED for status in counter])
        return all_deleted

    def _get_worker_status_counter(self):
        worker_counter = self._get_pod_counter(
            self._job_nodes[NodeType.WORKER]
        )
        evaluator_counter = self._get_pod_counter(
            self._job_nodes[NodeType.EVALUATOR]
        )
        tf_master_counter = self._get_pod_counter(
            self._job_nodes[NodeType.TF_MASTER]
        )
        counter = worker_counter + evaluator_counter + tf_master_counter
        return counter

    def _get_pod_counter(self, nodes):
        with self._lock:
            return Counter([node.status for node in nodes.values()])

    def all_critical_node_completed(self):
        alive_critical_nodes = []
        for _, nodes in self._job_nodes.items():
            for node_id, node in nodes.items():
                if node.critical and node.status in [
                    NodeStatus.INITIAL,
                    NodeStatus.PENDING,
                    NodeStatus.RUNNING,
                ]:
                    alive_critical_nodes.append(node_id)

        print(alive_critical_nodes)
        completed = not alive_critical_nodes
        if not completed:
            logger.info("Critical pods %s are running.", alive_critical_nodes)
        return completed

    def remove_worker(self, worker_id):
        if self._job_nodes[NodeType.WORKER][worker_id].critical:
            logger.info("Skip the critical worker %s", worker_id)
        else:
            # TODO: implement to delete a worker.
            logger.info("Delete worker %s", worker_id)

    def get_all_training_nodes(self):
        workers = self.get_running_workers()
        ps = self.get_cur_cluster_ps()
        workers.extend(ps)
        return workers

    def get_running_workers(self):
        """Return running worker, master and evaluator"""
        running_workers = []
        with self._lock:
            for node in self._job_nodes[NodeType.WORKER].values():
                if node.status == NodeStatus.RUNNING:
                    running_workers.append(node)

            for node in self._job_nodes[NodeType.TF_MASTER].values():
                if node.status == NodeStatus.RUNNING:
                    running_workers.append(node)

            for node in self._job_nodes[NodeType.EVALUATOR].values():
                if node.status == NodeStatus.RUNNING:
                    running_workers.append(node)
        return running_workers

    def stop(self):
        self._relaunch_pod = False
        with self._lock:
            for node_type in self._job_nodes.keys():
                for node in self._job_nodes[node_type].values():
                    node.critical = False
                    node.is_released = True
        self._stop_monitor = True

    def update_node_resource_usage(self, node_type, node_id, cpu, memory):
        node = self._job_nodes[node_type][node_id]
        node.update_resource_usage(cpu, memory)

    # TODO: implement the function.
    def start_auto_scale(self):
        """Start to auto scale"""
        pass

    # TODO: implement the function.
    def get_cur_cluster_ps(self):
        """Get PS nodes in the current training cluster."""
        return []

    # TODO: implement the function.
    def get_next_cluster_ps(self):
        """Get PS nodes in the next training cluster."""
        return []

    # TODO: implement the function.
    def ready_for_new_ps_cluster(self):
        return True

    # TODO: implement the function.
    def remove_training_nodes(self):
        """Remove all PS and workers"""
        pass


def create_node_manager(args, speed_monitor) -> NodeManager:
    # relaunch on worker failure for PS or custom strategy
    if (
        args.distribution_strategy != DistributionStrategy.PARAMETER_SERVER
        and args.distribution_strategy != DistributionStrategy.CUSTOM
    ):
        args.relaunch_on_worker_failure = 0

    job_resource = JobResourceConfig()
    job_resource.add_node_group_resource(
        NodeType.WORKER,
        args.num_workers,
        args.worker_resource_request,
        args.worker_pod_priority,
    )
    job_resource.add_node_group_resource(
        NodeType.PS,
        args.num_ps_pods,
        args.ps_resource_request,
        args.ps_pod_priority,
    )
    # Keep the same as the worker.
    evaluator_pod_priority = (
        args.evaluator_pod_priority
        if args.evaluator_pod_priority == "low"
        else "high"
    )
    job_resource.add_node_group_resource(
        NodeType.EVALUATOR,
        args.num_evaluators,
        args.evaluator_resource_request,
        evaluator_pod_priority,
    )
    job_resource.add_node_group_resource(
        NodeType.TF_MASTER,
        args.num_tf_master,
        args.tf_master_resource_request,
        args.tf_master_pod_priority,
    )
    critical_worker_index = get_critical_worker_index(args)
    # Custom distribution strategy does not exit if there are pending pods
    wait_pending_relaunch = (
        args.distribution_strategy == DistributionStrategy.CUSTOM
    )

    return NodeManager(
        job_resource=job_resource,
        job_name=args.job_name,
        namespace=args.namespace,
        relaunch_on_worker_failure=args.relaunch_on_worker_failure,
        ps_is_critical=args.ps_is_critical,
        critical_worker_index=critical_worker_index,
        wait_pending_relaunch=wait_pending_relaunch,
        ps_relaunch_max_num=args.ps_relaunch_max_num,
        use_ddp=args.use_ddp,
        speed_monitor=speed_monitor,
    )
