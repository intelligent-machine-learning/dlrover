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
import threading
import time
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
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.event_callback import (
    ClusterContext,
    NodeEventCallback,
)
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.node.status_flow import (
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node.training_node import (
    get_critical_worker_index,
    set_critical_node,
)
from dlrover.python.master.node.worker import (
    ChiefManager,
    EvaluatorManager,
    WorkerManager,
)
from dlrover.python.master.resource.job import (
    JobResource,
    JobResourceOptimizer,
)
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.master.scaler.factory import new_job_scaler
from dlrover.python.master.watcher.base_watcher import NodeEvent
from dlrover.python.master.watcher.factory import new_node_watcher
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.scheduler.job import JobArgs

_MAX_POD_RELAUNCH_COUNT = 5
_dlrover_context = Context.singleton_instance()


class NodeManager(object):
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
            job_args.distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        ):
            self._ps_is_critical = (
                job_args.node_args[NodeType.PS].critical_nodes == "all"
            )

        worker_restart_count = node_restart_count.get(NodeType.WORKER, 0)
        ps_restart_count = node_restart_count.get(NodeType.PS, 0)

        self._relaunch_on_worker_failure = min(
            worker_restart_count, _MAX_POD_RELAUNCH_COUNT
        )
        self._wait_pending_relaunch = wait_pending_relaunch
        self._start_launch_waiting_workers_time = time.time()
        self._stop_launch_worker_for_ps = False
        self._critical_worker_index = critical_worker_index
        self._ps_relaunch_max_num = min(
            ps_restart_count, _MAX_POD_RELAUNCH_COUNT
        )
        self._use_ddp = job_args.use_ddp
        self._node_event_callbacks: List[NodeEventCallback] = []
        self._chief_started = False
        self._stop_monitor = False
        self._speed_monitor: SpeedMonitor = speed_monitor

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()
        self._job_nodes: Dict[str, Dict[int, Node]] = {}

        self._elastic_job = job
        self._node_watcher = node_watcher
        self._scaler: Scaler = job_scaler
        self._job_optimizer = JobResourceOptimizer(
            self._job_resource.node_group_resources[NodeType.WORKER],
            self._job_resource.node_group_resources[NodeType.PS],
            job_args.scaling_optimizer,
            job_args.job_uuid,
            job_args.resource_limits,
        )
        self._init_training_node_manager()

    def start(self):
        self._job_optimizer.update_job_uuid(self._job_args.job_uuid)
        self._job_optimizer.init_job_resource(self._job_resource)
        self._adjust_worker_for_estimator()
        self._init_job_nodes()
        plan = self._create_initial_scale_plan()
        self._scaler.scale(plan)
        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()

    def _adjust_worker_for_estimator(self):
        if (
            self._job_args.distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        ):
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
            self._ps_relaunch_max_num,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._worker_manager = WorkerManager(
            self._job_nodes.get(NodeType.WORKER, {}),
            self._job_resource,
            self._ps_relaunch_max_num,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._evaluator_manager = EvaluatorManager(
            self._job_nodes.get(NodeType.EVALUATOR, {}),
            self._job_resource,
            self._ps_relaunch_max_num,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

    def add_node_event_callback(self, node_event_callback):
        self._node_event_callbacks.append(node_event_callback)

    def _init_job_nodes(self):
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
            except Exception as e:
                logger.warning(e)
                time.sleep(30)

    def _process_list_nodes(self, nodes: List[Node]):
        """Callback with node list by the list api of k8s."""
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
            "%s status change: %s to %s, by evt_type %s, phase %s",
            cur_node.name,
            old_status,
            new_status,
            event.event_type,
            new_status,
        )

        if should_relaunch:
            self._relaunch_node(cur_node)

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
            and self._enable_relaunch_node
            and node.relaunchable
        )
        if should_relaunch:
            if node.exit_reason == NodeExitReason.FATAL_ERROR:
                should_relaunch = False
            elif node.exit_reason == NodeExitReason.OOM:
                mem = node.config_resource.memory
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
                    if node.type == NodeType.PS:
                        self._job_optimizer.adjust_oom_ps_resource(node)
                    else:
                        self._job_optimizer.adjust_oom_worker_resource(node)
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
        logger.info("Relaunch node: {}".format(node.name))
        if node.type == NodeType.WORKER:
            plan = self._worker_manager.relaunch_node(node)
        elif node.type == NodeType.PS:
            plan = self._ps_manager.relaunch_node(node)
        elif node.type == NodeType.EVALUATOR:
            plan = self._evaluator_manager.relaunch_node(node)
        elif node.type == NodeType.CHIEF:
            plan = self._chief_manager.relaunch_node(node)
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
            logger.info("plan %s", plan)

    def get_running_nodes(self):
        nodes = self._chief_manager.get_running_nodes()
        nodes.extend(self._worker_manager.get_running_nodes())
        nodes.extend(self._evaluator_manager.get_running_nodes())
        nodes.extend(self._ps_manager.get_training_ps_cluster())
        return nodes

    def get_running_workers(self):
        return self._worker_manager.get_running_nodes()

    def post_ps_ready(self):
        self._ps_manager.process_after_ps_cluster_ready()

    def stop(self):
        self._enable_relaunch_node = False
        with self._lock:
            for node_type in self._job_nodes.keys():
                for node in self._job_nodes[node_type].values():
                    node.critical = False
                    node.is_released = True
                    node.relaunchable = False
        self._stop_monitor = True

    def update_node_resource_usage(self, node_type, node_id, cpu, memory):
        node = self._job_nodes[node_type][node_id]
        node.update_resource_usage(cpu, memory)

    def get_cur_cluster_ps(self):
        """Get PS nodes in the current training cluster."""
        return self._ps_manager.get_training_ps_cluster()

    def get_next_cluster_ps(self):
        """Get PS nodes in the next training cluster."""
        return self._ps_manager.get_next_training_ps_cluster()

    def ready_for_new_ps_cluster(self):
        return self._ps_manager.get_ready_for_new_ps_cluster()

    def remove_training_nodes(self):
        """Remove all PS and workers"""
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
                plan.remove_nodes.append(node.name)
        self._scaler.scale(plan)

    def start_auto_scale(self):
        """Start to auto-scale nodes to improve the training throughput."""
        if not self._chief_started:
            logger.info("Chief started!")
            self._chief_started = True
            if self._job_resource.worker_num > 1:
                plan = self._job_optimizer.optimize_worker_resource()
                self._execute_job_optimization_plan(plan)
            if (
                not _dlrover_context.auto_ps_enabled
                and not _dlrover_context.auto_worker_enabled
            ):
                return
            if self._speed_monitor:
                self._speed_monitor.set_target_worker_num(
                    self._job_resource.worker_num
                    + self._job_resource.chief_num
                )
                threading.Thread(
                    target=self._periodic_optimize_running_resource,
                    name="resource-autoscaler",
                    daemon=True,
                ).start()

    def _periodic_optimize_running_resource(self):
        """Adjust job resource periodically and stop adjustment
        if there is a failed worker with the fatal error.
        """
        logger.info("Start to auto scale ")
        last_plan_time = 0
        opt_interval = _dlrover_context.seconds_interval_to_optimize
        while True:
            if self._stop_launch_worker_for_ps:
                logger.info("Stop to autoscale the number of worker.")
                break
            if (
                self._speed_monitor.worker_adjustment_finished()
                # Control the interval to query plans
                and time.time() - last_plan_time > opt_interval
                and not self._ps_manager.exist_migrated_ps_nodes()
            ):
                plan = self._job_optimizer.get_job_resource_plan()
                if plan:
                    last_plan_time = time.time()
                self._execute_job_optimization_plan(plan)
            time.sleep(30)

    def _execute_job_optimization_plan(self, plan: ResourcePlan):
        """Execute the optimization plan of the training job.
        The plan may adjust the number of PS and workers or
        adjust the cpu and memory of nodes.
        """
        scale_plan = ScalePlan()
        if plan.empty():
            return scale_plan
        for node_type, group in plan.node_group_resources.items():
            if group.count > 0:
                self._job_resource.update_node_group_resource(
                    node_type,
                    group.count,
                    group.node_resource.cpu,
                    group.node_resource.memory,
                )
                scale_plan.node_group_resources[node_type] = copy.deepcopy(
                    self._job_resource.get_node_group_resource(node_type)
                )
                if node_type == NodeType.PS:
                    self._ps_manager.adjust_ps(group)
                elif node_type == NodeType.WORKER:
                    self._speed_monitor.set_target_worker_num(group.count)
                    self._worker_manager.adjust_worker(group)
        if len(plan.node_resources) > 0:
            migration_plan = self._migrate_nodes(plan.node_resources)
            scale_plan.merge(migration_plan)
        self._set_ps_addrs_in_plan(scale_plan)
        self._scaler.scale(scale_plan)
        return scale_plan

    def _migrate_nodes(self, node_resources):
        workers: Dict[str, NodeResource] = {}
        ps: Dict[str, NodeResource] = {}
        for name, resource in node_resources.items():
            type = name.split("-")[-2]
            if type == NodeType.WORKER:
                workers[name] = resource
            elif type == NodeType.PS:
                ps[name] = resource

        scale_plan = ScalePlan()
        if len(ps) > 0:
            plan = self._ps_manager.migrate_parameter_servers(ps)
            scale_plan.merge(plan)
            self._speed_monitor.reset_running_speed_monitor()

        if len(workers) > 0:
            plan = self._worker_manager.migrate_workers(workers)
            scale_plan.merge(plan)
        return scale_plan

    def _set_ps_addrs_in_plan(self, plan: ScalePlan):
        ps_addrs = self._ps_manager.get_ps_addrs()
        plan.ps_addrs.extend(ps_addrs)

    def cut_timeout_pending_node_cpu(self):
        """Cut down CPU cores of pending pod at the job starts"""
        if self._chief_started:
            return
        if _dlrover_context.auto_ps_enabled:
            self._ps_manager.cut_pending_node_cpu()
        if _dlrover_context.auto_worker_enabled:
            self._worker_manager.cut_pending_node_cpu()


def create_node_manager(args: JobArgs, speed_monitor) -> NodeManager:
    # relaunch on worker failure for PS or custom strategy
    if (
        args.distribution_strategy != DistributionStrategy.PARAMETER_SERVER
        and args.distribution_strategy != DistributionStrategy.CUSTOM
    ):
        args.node_args[NodeType.WORKER].restart_count = 0

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

    return NodeManager(
        job_args=args,
        critical_worker_index=critical_worker_index,
        wait_pending_relaunch=wait_pending_relaunch,
        speed_monitor=speed_monitor,
        job=elastic_job,
        node_watcher=node_watcher,
        job_scaler=job_scaler,
    )
