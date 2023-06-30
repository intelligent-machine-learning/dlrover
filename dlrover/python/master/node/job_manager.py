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
import os
import threading
import time
import traceback
from typing import Dict, List

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeEventType,
    NodeExitReason,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.event_callback import (
    ClusterContext,
    NodeEventCallback,
)
from dlrover.python.master.node.job_auto_scaler import (
    JobAutoScaler,
    new_job_auto_scaler,
)
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.node.status_flow import (
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node.training_node import (
    get_critical_worker_index,
    set_critical_node,
    update_nodes_priority,
)
from dlrover.python.master.node.worker import (
    ChiefManager,
    EvaluatorManager,
    WorkerManager,
)
from dlrover.python.master.resource.job import (
    AllreduceJobResourceOptimizer,
    JobResource,
    JobResourceOptimizer,
    PSJobResourceOptimizer,
)
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.master.scaler.factory import new_job_scaler
from dlrover.python.master.watcher.base_watcher import NodeEvent
from dlrover.python.master.watcher.factory import (
    new_node_watcher,
    new_scale_plan_watcher,
)
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.scheduler.job import ElasticJob, JobArgs

_dlrover_context = Context.singleton_instance()

_MAX_POD_RELAUNCH_COUNT = 5


class JobManager(object):
    def __init__(
        self,
        job_args: JobArgs,
        critical_worker_index={},
        wait_pending_relaunch=False,
        speed_monitor=None,
        job=None,
        node_watcher=None,
        job_scaler=None,
    ):
        self._job_resource = JobResource()
        node_restart_count: Dict[str, int] = {}
        for type, node_args in job_args.node_args.items():
            self._job_resource.node_group_resources[
                type
            ] = node_args.group_resource
            node_restart_count[type] = node_args.restart_count

        self._job_args = job_args
        self._ps_is_critical = False
        if (
            job_args.distribution_strategy == DistributionStrategy.PS
            or job_args.distribution_strategy == DistributionStrategy.CUSTOM
        ):
            self._ps_is_critical = (
                job_args.node_args[NodeType.PS].critical_nodes == "all"
            )
            self._job_optimizer: JobResourceOptimizer = PSJobResourceOptimizer(
                self._job_resource.node_group_resources[NodeType.WORKER],
                self._job_resource.node_group_resources[NodeType.PS],
                job_args.optimize_mode,
                job_args.job_uuid,
                job_args.resource_limits,
            )
        elif job_args.distribution_strategy == DistributionStrategy.ALLREDUCE:
            self._job_optimizer: JobResourceOptimizer = (
                AllreduceJobResourceOptimizer(
                    self._job_resource.node_group_resources[NodeType.WORKER],
                    job_args.job_uuid,
                )
            )
        else:
            raise ValueError(
                f"Distribution strategy {job_args.distribution_strategy} "
                "is not supported. You can specify it with "
                "ParameterServerStrategy/AllreduceStrategy."
            )
        logger.info("New job optimizer : %s", self._job_optimizer.__class__)

        worker_restart_count = node_restart_count.get(NodeType.WORKER, 0)
        ps_restart_count = node_restart_count.get(NodeType.PS, 0)

        self._relaunch_on_worker_failure = min(
            worker_restart_count, _MAX_POD_RELAUNCH_COUNT
        )
        self._wait_pending_relaunch = wait_pending_relaunch
        self._start_launch_waiting_workers_time = time.time()
        self._critical_worker_index = critical_worker_index
        self._ps_relaunch_max_num = min(
            ps_restart_count, _MAX_POD_RELAUNCH_COUNT
        )
        self._node_event_callbacks: List[NodeEventCallback] = []
        self._stop_monitor = False
        self._speed_monitor: SpeedMonitor = speed_monitor

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()
        self._job_nodes: Dict[str, Dict[int, Node]] = {}

        self._elastic_job: ElasticJob = job
        self._node_watcher = node_watcher

        self._scaler_watcher = new_scale_plan_watcher(
            job_args.platform,
            job_args.job_name,
            job_args.namespace,
            job_args.job_uuid,
        )
        self._scaler: Scaler = job_scaler
        self._init_training_node_manager()

    def start(self):
        self._job_optimizer.update_job_uuid(self._job_args.job_uuid)
        self._job_optimizer.init_job_resource(self._job_resource)
        self._adjust_worker_for_estimator()
        self._init_nodes()
        self._init_job_auto_scaler()
        plan = self._create_initial_scale_plan()
        self._scaler.scale(plan)
        worker_num = 0
        if NodeType.WORKER in plan.node_group_resources:
            worker_num = plan.node_group_resources[NodeType.WORKER].count
        if NodeType.CHIEF in plan.node_group_resources:
            worker_num += plan.node_group_resources[NodeType.CHIEF].count
        self._speed_monitor.set_target_worker_num(worker_num)
        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()
        threading.Thread(
            target=self._monitor_scale_plan_crd,
            name="scaleplan_monitor",
            daemon=True,
        ).start()

    def _adjust_worker_for_estimator(self):
        if self._job_args.distribution_strategy == DistributionStrategy.PS:
            self._job_resource.adjust_worker_for_estimator()

    def _create_initial_scale_plan(self):
        scale_plan = ScalePlan()
        scale_plan.node_group_resources = copy.deepcopy(
            self._job_resource.node_group_resources
        )
        scale_plan.ps_addrs = self._ps_manager.get_ps_addrs()
        return scale_plan

    def _init_training_node_manager(self):
        self._ps_manager = ParameterServerManager(
            self._job_nodes.get(NodeType.PS, {}),
            self._job_resource,
            self._ps_relaunch_max_num,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._chief_manager = ChiefManager(
            self._job_nodes.get(NodeType.CHIEF, {}),
            self._job_resource,
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._worker_manager = WorkerManager(
            self._job_nodes.get(NodeType.WORKER, {}),
            self._job_resource,
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._evaluator_manager = EvaluatorManager(
            self._job_nodes.get(NodeType.EVALUATOR, {}),
            self._job_resource,
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

    def add_node_event_callback(self, node_event_callback):
        self._node_event_callbacks.append(node_event_callback)

    def _init_nodes(self):
        self._job_nodes = self._job_resource.init_job_node_meta(
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

        # worker and eval ids for nodes that should be created
        # after all ps are running.
        self._workers_waiting_ps_running = []

        self._enable_relaunch_node = True
        self._pending_relaunch_count = 0

        set_critical_node(
            self._job_nodes,
            self._ps_is_critical,
            self._ps_relaunch_max_num,
            self._critical_worker_index,
        )
        update_nodes_priority(self._job_nodes)

        self._ps_manager.update_nodes(self._job_nodes.get(NodeType.PS, {}))
        self._chief_manager.update_nodes(
            self._job_nodes.get(NodeType.CHIEF, {})
        )
        self._worker_manager.update_nodes(
            self._job_nodes.get(NodeType.WORKER, {})
        )
        self._evaluator_manager.update_nodes(
            self._job_nodes.get(NodeType.EVALUATOR, {})
        )

    def _init_job_auto_scaler(self):
        self._job_autoscaler: JobAutoScaler = new_job_auto_scaler(
            self._job_args.distribution_strategy,
            self._job_resource,
            self._job_nodes,
            self._job_optimizer,
            self._speed_monitor,
            self._ps_manager,
            self._worker_manager,
            self._scaler,
        )
        logger.info(
            "Create job autoscaler: %s", self._job_autoscaler.__class__
        )

    def _monitor_nodes(self):
        logger.info("Start to monitor nodes")
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
                        detail_trace_back = traceback.format_exc()
                        logger.warning(detail_trace_back)
            except Exception as e:
                logger.warning(e)
                time.sleep(30)

    def _monitor_scale_plan_crd(self):
        """Monitor the Scaler CRD from users to adjust the job resource"""
        logger.info("Start to monitor Scaler CRD")
        while True:
            try:
                if self._stop_monitor:
                    logger.info("Stop monitoring Scaler CRDs")
                    break
                for plan in self._scaler_watcher.watch():
                    try:
                        self._job_autoscaler.execute_job_optimization_plan(
                            plan
                        )
                    except Exception as e:
                        logger.warning(e)
                        detail_trace_back = traceback.format_exc()
                        logger.warning(detail_trace_back)
            except Exception as e:
                logger.warning(e)
                detail_trace_back = traceback.format_exc()
                logger.warning(detail_trace_back)
                time.sleep(5)

    def _process_list_nodes(self, nodes: List[Node]):
        """Callback with node list by the list api of k8s."""
        if not nodes:
            return
        exist_nodes: Dict[str, List[int]] = {}
        for node_type in self._job_nodes.keys():
            exist_nodes[node_type] = []
        for node in nodes:
            exist_nodes[node.type].append(node.id)
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
                    and node_id not in exist_nodes[node_type]
                ):
                    logger.info(
                        "Node %s %s is deleted without the event",
                        node_type,
                        node_id,
                    )
                    node.is_released = True

    def close_job(self):
        plan = ScalePlan()
        ps_resource = NodeGroupResource.new_empty()
        worker_reource = NodeGroupResource.new_empty()
        plan.node_group_resources = {
            "worker": worker_reource,
            "ps": ps_resource,
        }
        self._scaler.scale(plan=plan)
        os._exit(0)

    def _process_event(self, event: NodeEvent):
        node_type = event.node.type
        node_id = event.node.id
        if node_id not in self._job_nodes[node_type]:
            self._job_nodes[node_type][node_id] = event.node
            return
        else:
            cur_node = self._job_nodes[node_type][node_id]
            cur_node.update_info(
                name=event.node.name,
                start_time=event.node.start_time,
                create_time=event.node.create_time,
            )

        # For the given node id, check whether it meets
        # the state change condition
        if event.event_type == "exit":
            self.close_job()
        new_status = event.node.status
        with self._lock:
            old_status = cur_node.status
            status_change_flow: NodeStateFlow = get_node_state_flow(
                old_status, event.event_type, new_status
            )
            cur_node.update_status(new_status)
            # If there is no matched state change, return directly
            # If the node has been succeed, return directly
            if (
                status_change_flow is None
                or status_change_flow.from_status == NodeStatus.SUCCEEDED
            ):
                return

            # Update the node status
            new_status = status_change_flow.to_status
            cur_node.set_exit_reason(event.node.exit_reason)
            self._process_node_events(status_change_flow, cur_node)

            should_relaunch = self._should_relaunch(
                cur_node, status_change_flow
            )
            if should_relaunch and self._wait_pending_relaunch:
                self._pending_relaunch_count += 1

        logger.info(
            "%s status change: %s to %s, by evt_type %s reason %s",
            cur_node.name,
            old_status,
            new_status,
            event.event_type,
            cur_node.exit_reason,
        )

        if should_relaunch:
            self._relaunch_node(cur_node)

    def _process_node_events(
        self, status_change_flow: NodeStateFlow, node: Node
    ):
        cluster_context = ClusterContext(job_manager=self)
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
            and self._enable_relaunch_node
            and node.relaunchable
        )
        if should_relaunch:
            if (
                node.exit_reason == NodeExitReason.FATAL_ERROR
                and not _dlrover_context.relaunch_error
            ):
                should_relaunch = False
            elif node.exit_reason == NodeExitReason.OOM:
                mem = node.config_resource.memory
                if mem >= NodeResourceLimit.MAX_MEMORY:
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
                    self._job_optimizer.adjust_oom_resource(node)
            elif node.exit_reason != NodeExitReason.KILLED:
                if node.relaunch_count > node.max_relaunch_count:
                    logger.warning(
                        "The relaunch count for Error has been exhausted."
                    )
                    should_relaunch = False
        if should_relaunch:
            node.inc_relaunch_count()

        return should_relaunch

    def _relaunch_node(self, node: Node):
        if node.type == NodeType.WORKER:
            plan = self._worker_manager.relaunch_node(node)
        elif node.type == NodeType.PS:
            plan = self._ps_manager.relaunch_node(node)
        elif node.type == NodeType.EVALUATOR:
            plan = self._evaluator_manager.relaunch_node(node)
        elif node.type == NodeType.CHIEF or node.type == NodeType.MASTER:
            plan = self._chief_manager.relaunch_node(node)
        else:
            logger.error("Not support node type %s", node.type)
        logger.info("Relaunch node: {}".format(node.name))
        self._set_ps_addrs_in_plan(plan)
        self._scaler.scale(plan)

    def all_workers_exited(self):
        return (
            self._chief_manager.all_nodes_exited()
            and self._worker_manager.all_nodes_exited()
            and self._evaluator_manager.all_nodes_exited()
        )

    def all_workers_failed(self):
        return (
            self._chief_manager.all_nodes_failed()
            and self._worker_manager.all_nodes_failed()
            and self._evaluator_manager.all_nodes_failed()
        )

    def all_workers_deleted(self):
        return (
            self._chief_manager.all_nodes_deleted()
            and self._worker_manager.all_nodes_deleted()
            and self._evaluator_manager.all_nodes_deleted()
        )

    def all_critical_node_completed(self):
        alive_critical_nodes = []
        for _, nodes in self._job_nodes.items():
            for node in nodes.values():
                if node.critical and node.status in [
                    NodeStatus.INITIAL,
                    NodeStatus.PENDING,
                    NodeStatus.RUNNING,
                ]:
                    alive_critical_nodes.append(node.name)

        completed = not alive_critical_nodes
        if not completed:
            logger.info("Critical nodes %s are running.", alive_critical_nodes)
        return completed

    def remove_worker(self, worker_id):
        if self._job_nodes[NodeType.WORKER][worker_id].critical:
            logger.info("Skip the critical worker %s", worker_id)
        else:
            logger.info("Delete worker %s", worker_id)
            plan = self._worker_manager.remove_node(worker_id)
            logger.info("plan %s", plan.toJSON())
            self._scaler.scale(plan)

    def get_running_nodes(self):
        nodes = self._chief_manager.get_running_nodes()
        nodes.extend(self._worker_manager.get_running_nodes())
        nodes.extend(self._evaluator_manager.get_running_nodes())
        nodes.extend(self._ps_manager.get_training_ps_cluster())
        return nodes

    def get_running_workers(self):
        workers = self._worker_manager.get_running_nodes()
        chiefs = self._chief_manager.get_running_nodes()
        workers.extend(chiefs)
        return workers

    def post_ps_ready(self):
        plan = self._ps_manager.process_after_ps_cluster_ready()
        if not plan.empty():
            self._scaler.scale(plan)
        else:
            logger.info("Skip an empty scaleplan")

    def stop(self):
        self._enable_relaunch_node = False
        with self._lock:
            for node_type in self._job_nodes.keys():
                for node in self._job_nodes[node_type].values():
                    node.critical = False
                    node.is_released = True
                    node.relaunchable = False
            for node in self._job_nodes[NodeType.WORKER].values():
                node.eval_time = self._speed_monitor.get_worker_eval_time(
                    node.id
                )
        self._stop_monitor = True

    def update_node_resource_usage(self, node_type, node_id, cpu, memory):
        node = self._job_nodes[node_type][node_id]
        node.update_resource_usage(cpu, memory)

    def update_node_service_addr(self, node_type, node_id, service_addr):
        node = self._job_nodes[node_type][node_id]
        node.update_service_address(service_addr)
        node.status = NodeStatus.RUNNING
        node.is_released = False
        self._job_nodes[node_type][node_id] = node

    def get_cur_cluster_ps(self):
        """Get PS nodes in the current training cluster."""
        logger.info("job nodes are {}".format(self._job_nodes))
        return self._ps_manager.get_training_ps_cluster()

    def get_next_cluster_ps(self):
        """Get PS nodes in the next training cluster."""
        return self._ps_manager.get_next_training_ps_cluster()

    def ready_for_new_ps_cluster(self):
        """Check whether ps cluster is used to training"""
        return self._ps_manager.get_ready_for_new_ps_cluster()

    def has_ps_failure(self):
        """Check whether ther is PS failure"""
        return self._ps_manager.has_ps_failure()

    def remove_training_nodes(self):
        """Remove all PS and workers"""
        self._job_autoscaler.stop_auto_scaling()
        plan = ScalePlan()
        training_nodes = list(
            self._job_nodes[NodeType.WORKER].values()
        ) + list(self._job_nodes[NodeType.PS].values())
        for node in training_nodes:
            if (
                node.status in [NodeStatus.RUNNING, NodeStatus.PENDING]
                and not node.is_released
            ):
                node.critical = False
                node.relaunchable = False
                node.is_released = True
                node.status = NodeStatus.DELETED
                logger.info("Remove node %s", node.name)
                plan.remove_nodes.append(node)
        self._scaler.scale(plan)

    def start_auto_scaling(self):
        """Start to auto-scale nodes to improve the training throughput."""
        self._job_autoscaler.start_auto_scaling()

    def _set_ps_addrs_in_plan(self, plan: ScalePlan):
        ps_addrs = self._ps_manager.get_ps_addrs()
        plan.ps_addrs.extend(ps_addrs)

    def all_running_node_hanged(self):
        node_hang = self._worker_manager.running_nodes_hanged()
        node_hang.extend(self._chief_manager.running_nodes_hanged())
        node_hang.extend(self._evaluator_manager.running_nodes_hanged())
        node_hang.extend(self._ps_manager.running_nodes_hanged())
        if node_hang:
            return all(node_hang)
        return False

    def remove_not_participated_workers(self, workers):
        plan = self._worker_manager.remove_not_participated_workers(workers)
        self._scaler.scale(plan)

    def pend_without_workers(self):
        """Check whether to wait for evicted workers."""
        if self._worker_manager.has_failed_worker():
            return False
        elif self._worker_manager.wait_worker_restart():
            return True
        else:
            return False

    def remove_breakdown_node(self, node_type, node_id):
        node = self._job_nodes[node_type][node_id]
        logger.warning(f"Node {node.name} is breakdown.")


def create_job_manager(args: JobArgs, speed_monitor) -> JobManager:
    critical_worker_index = get_critical_worker_index(args)
    # Custom distribution strategy does not exit if there are pending nodes
    wait_pending_relaunch = (
        args.distribution_strategy == DistributionStrategy.CUSTOM
    )

    elastic_job = new_elastic_job(args.platform, args.job_name, args.namespace)
    node_watcher = new_node_watcher(
        args.platform, args.job_name, args.namespace
    )
    job_scaler = new_job_scaler(args.platform, args.job_name, args.namespace)

    return JobManager(
        job_args=args,
        critical_worker_index=critical_worker_index,
        wait_pending_relaunch=wait_pending_relaunch,
        speed_monitor=speed_monitor,
        job=elastic_job,
        node_watcher=node_watcher,
        job_scaler=job_scaler,
    )
