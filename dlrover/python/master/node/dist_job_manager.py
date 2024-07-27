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
from datetime import datetime
from typing import Dict, List, Optional

from dlrover.python.common.constants import (
    DistributionStrategy,
    JobExitReason,
    NodeEventType,
    NodeExitReason,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
    TrainingExceptionLevel,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.grpc import ParallelConfig
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeGroupResource
from dlrover.python.master.monitor.error_monitor import K8sJobErrorMonitor
from dlrover.python.master.node.event_callback import (
    ClusterContext,
    NodeEventCallback,
)
from dlrover.python.master.node.job_auto_scaler import (
    JobAutoScaler,
    new_job_auto_scaler,
)
from dlrover.python.master.node.job_manager import JobManager
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
    JobResourceOptimizer,
    PSJobResourceOptimizer,
)
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.master.scaler.factory import new_job_scaler
from dlrover.python.master.watcher.base_watcher import NodeEvent, NodeWatcher
from dlrover.python.master.watcher.factory import (
    new_node_watcher,
    new_scale_plan_watcher,
)
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.scheduler.job import ElasticJob, JobArgs

_dlrover_context = Context.singleton_instance()

_MAX_POD_RELAUNCH_COUNT = 5


class DistributedJobManager(JobManager):
    """DistributedJobManager manages the nodes of a distributed job on
    a k8s cluster. For a job, the manager will:
    - create nodes on a cluster.
    - monitor nodes on a cluster.
    - scale up/down nodes on a cluster.
    """

    def __init__(
        self,
        job_args: JobArgs,
        critical_worker_index={},
        wait_pending_relaunch=False,
        speed_monitor=None,
        job=None,
        node_watcher: Optional[NodeWatcher] = None,
        job_scaler=None,
        error_monitor=None,
    ):
        super().__init__(job_args, speed_monitor, error_monitor)
        self._remove_exited_node = job_args.remove_exited_node
        node_restart_count: Dict[str, int] = {}
        for type, node_args in job_args.node_args.items():
            self._job_resource.node_group_resources[
                type
            ] = node_args.group_resource
            node_restart_count[type] = node_args.restart_count

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
            self._job_optimizer = AllreduceJobResourceOptimizer(
                self._job_resource.node_group_resources[NodeType.WORKER],
                job_args.job_uuid,
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

        # Protects followed variables, which are accessed from event_cb.
        self._lock = threading.Lock()

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
        self._scaler.start()
        self._job_optimizer.update_job_uuid(self._job_args.job_uuid)
        self._job_optimizer.init_job_resource(self._job_resource)
        self._adjust_worker_for_estimator()
        self._init_nodes()
        self._init_job_auto_scaler()
        plan = self._create_initial_scale_plan()
        if not self._has_running_workers():
            # The the job relaunches the evicted master, there are alive
            # worker nodes and the master does not need to launch workers.
            logger.info(
                "The newly master starts launching workers at beginning."
            )
            self._scaler.scale(plan)
        else:
            logger.info(
                "The recovered master skips launching workers at beginning."
            )
        worker_num = 0
        if NodeType.WORKER in plan.node_group_resources:
            worker_num = plan.node_group_resources[NodeType.WORKER].count
        if NodeType.CHIEF in plan.node_group_resources:
            worker_num += plan.node_group_resources[NodeType.CHIEF].count
        self._speed_monitor.set_target_worker_num(worker_num)
        self._training_node_configure.set_node_num(worker_num)
        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()
        threading.Thread(
            target=self._monitor_node_heart_beat,
            name="node_heart_beat_monitor",
            daemon=True,
        ).start()
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            threading.Thread(
                target=self._monitor_scale_plan_crd,
                name="scaleplan_monitor",
                daemon=True,
            ).start()

    def _has_running_workers(self):
        nodes = self._node_watcher.list()
        for node in nodes:
            if node.status in [NodeStatus.PENDING, NodeStatus.RUNNING]:
                return True
        return False

    def get_worker_num(self):
        return self._job_resource.worker_num

    def should_early_stop(self):
        # ps pending judgement: any ps pod pending timeout
        timeout_ps_nodes = (
            self._ps_manager.get_pending_timeout_oom_recovered_node()
        )
        if len(timeout_ps_nodes) > 0:
            msg = (
                "Stop the training early because the nodes recovered from OOM "
                "are pending too long and have timed out."
            )
            self._error_monitor.process_error(
                timeout_ps_nodes[0],
                0,
                msg,
                level=TrainingExceptionLevel.ERROR,
            )
            return True, JobExitReason.PENDING_TIMEOUT, msg

        # worker pending judgement:
        if self._worker_manager.is_training_hang_by_pending():
            msg = (
                "Stop the training early because 1) there is node pending "
                "2) alive worker number consistently less than the min "
                "training nodes required 3) pending time last exceed limit."
            )
            self._error_monitor.process_error(
                None, 0, msg, level=TrainingExceptionLevel.ERROR
            )
            return True, JobExitReason.PENDING_TIMEOUT, msg

        # insufficient worker judgement
        if self._worker_manager.is_training_hang_by_insufficient_worker():
            msg = (
                "Stop the training early because there isn't enough node to "
                "keep training."
            )
            self._error_monitor.process_error(
                None, 0, msg, level=TrainingExceptionLevel.ERROR
            )
            return True, JobExitReason.UNCOMPLETED_TIMEOUT, msg

        # no need to early stop
        return False, "", ""

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

        chief_nodes = self._job_nodes.get(NodeType.CHIEF, {})
        if not chief_nodes:
            chief_nodes = self._job_nodes.get(NodeType.MASTER, {})
        self._chief_manager = ChiefManager(
            chief_nodes,
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
        chief_nodes = self._job_nodes.get(NodeType.CHIEF, {})
        if not chief_nodes:
            chief_nodes = self._job_nodes.get(NodeType.MASTER, {})
        self._chief_manager.update_nodes(chief_nodes)
        workers = self._job_nodes.get(NodeType.WORKER, {})
        self._worker_manager.update_nodes(workers)
        evaluators = self._job_nodes.get(NodeType.EVALUATOR, {})
        self._evaluator_manager.update_nodes(evaluators)

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
        logger.info("Start monitoring nodes events.")
        while True:
            if self._stopped:
                logger.info("Stop monitoring nodes.")
                break
            try:
                nodes = self._node_watcher.list()
                self._process_list_nodes(nodes)
                if self._stopped:
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
            time.sleep(5)

    def _monitor_node_heart_beat(self):
        logger.info("Start monitoring the heartbeat of nodes.")
        while True:
            if self._stopped:
                logger.info("Stop monitoring the heartbeat of nodes.")
                break
            with self._lock:
                try:
                    events = self._get_dead_node_event()
                except Exception as e:
                    logger.warning(e)
                    events = []

            for event in events:
                try:
                    self._process_event(event)
                except Exception as e:
                    logger.warning(e)
                    detail_trace_back = traceback.format_exc()
                    logger.warning(detail_trace_back)
            time.sleep(15)

    def _get_dead_node_event(self, window_interval=600) -> List[NodeEvent]:
        now = time.time()
        dead_events: List[NodeEvent] = []
        logger.debug(f"Current job nodes are: {self._job_nodes}.")
        for _, nodes in self._job_nodes.items():
            for _, node in nodes.items():
                if (
                    node.heartbeat_time > 0
                    and now - node.heartbeat_time > window_interval
                    and node.start_time
                    and node.create_time
                    and node.status == NodeStatus.RUNNING
                ):
                    if (
                        node.heartbeat_time <= node.start_time.timestamp()
                        or node.heartbeat_time <= node.create_time.timestamp()
                    ):
                        logger.warning(
                            f"Skip dead node judgement for "
                            f"node: {node.id}-{node.name} "
                            f"because heartbeat time < create/start time. "
                            f"Current nodes: {self._get_nodes_time_info()}."
                        )
                        continue

                    event_node = copy.deepcopy(node)
                    event_node.status = NodeStatus.FAILED
                    event_node.exit_reason = NodeExitReason.NO_HEARTBEAT
                    event = NodeEvent(
                        event_type=NodeEventType.DELETED,
                        node=event_node,
                    )
                    dead_events.append(event)
                    error_data = (
                        f"No heartbeat for over {window_interval} seconds."
                    )
                    self._error_monitor.process_error(
                        node,
                        node.relaunch_count,
                        error_data,
                        TrainingExceptionLevel.NODE_ERROR,
                    )
                    logger.warning(
                        f"The node {node.id}-{node.name} has not sent a "
                        f"heartbeat for over {window_interval} seconds, "
                        f"last heartbeat: {node.heartbeat_time}, "
                        f"created at: {node.create_time}, "
                        f"started at: {node.start_time}."
                    )
        return dead_events

    def _get_nodes_time_info(self):
        result = {}
        for _, nodes in self._job_nodes.items():
            for _, node in nodes.items():
                if node.heartbeat_time == 0:
                    heartbeat_time = 0
                else:
                    heartbeat_time = datetime.fromtimestamp(
                        node.heartbeat_time
                    )
                result_dict = {
                    "name": node.name,
                    "type": node.type,
                    "create": node.create_time,
                    "start": node.start_time,
                    "heartbeat": heartbeat_time,
                }
                result[node.id] = result_dict

        return result

    def _monitor_scale_plan_crd(self):
        """Monitor the Scaler CRD from users to adjust the job resource"""
        logger.info("Start to monitor Scaler CRD")
        while True:
            try:
                if self._stopped:
                    logger.info("Stop monitoring Scaler CRDs.")
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
            #  Avoid dictionary keys changed during iteration
            type_nodes = list(self._job_nodes[node_type].values())
            for node in type_nodes:
                if (
                    node.status != NodeStatus.INITIAL
                    and not node.is_released
                    and node.id not in exist_nodes[node_type]
                ):
                    logger.info(
                        "Node %s %s is deleted without the event",
                        node_type,
                        node.id,
                    )
                    node.is_released = True
                    new_node = copy.deepcopy(node)
                    new_node.status = NodeStatus.DELETED
                    event = NodeEvent(NodeEventType.DELETED, new_node)
                    self._process_event(event)

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
            logger.info(f"The node {event.node.name} is released.")
            return
        else:
            cur_node = self._job_nodes[node_type][node_id]
            logger.debug(
                f"Update node({cur_node.id}), "
                f"name: {cur_node.name}->{event.node.name}, "
                f"start_time: {cur_node.start_time}"
                f"->{event.node.start_time}, "
                f"create_time: {cur_node.create_time}"
                f"->{event.node.create_time}, "
                f"host_name: {cur_node.host_name}"
                f"->{event.node.host_name},"
                f"host_ip: {cur_node.host_ip}"
                f"->{event.node.host_ip}, "
                f"restart_training: {cur_node.restart_training}"
                f"->{event.node.restart_training}, "
                f"relaunch_count: {cur_node.relaunch_count}"
                f"->{event.node.relaunch_count}"
            )
            cur_node.update_info(
                name=event.node.name,
                start_time=event.node.start_time,
                create_time=event.node.create_time,
                host_name=event.node.host_name,
                host_ip=event.node.host_ip,
                restart_training=event.node.restart_training,
                relaunch_count=event.node.relaunch_count,
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
            # If there is no matched state change, return directly
            # If the node has been succeed, return directly
            if (
                status_change_flow is None
                or status_change_flow.from_status == NodeStatus.SUCCEEDED
            ):
                return

            # Update the node status
            cur_node.update_status(new_status)
            new_status = status_change_flow.to_status
            cur_node.set_exit_reason(event.node.exit_reason)
            self._process_node_events(status_change_flow, cur_node)

            should_relaunch = self._should_relaunch(
                cur_node, status_change_flow
            )
            if should_relaunch and self._wait_pending_relaunch:
                self._pending_relaunch_count += 1

        msg = (
            f"{cur_node.name} status change: {old_status} to {new_status} "
            f"by the event {event.event_type}. "
        )
        if new_status in [NodeStatus.FAILED, NodeStatus.DELETED]:
            msg += f"Exit reason is {cur_node.exit_reason}"
        logger.info(msg)

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
                and not _dlrover_context.relaunch_always
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
                if node.relaunch_count >= node.max_relaunch_count:
                    logger.warning(
                        "The relaunch count "
                        f"{node.relaunch_count}/{node.max_relaunch_count} "
                        "has been exhausted."
                    )
                    should_relaunch = False
        if should_relaunch:
            node.relaunch_count += 1

        return should_relaunch

    def _relaunch_node(self, node: Node):
        if node.type == NodeType.WORKER:
            plan = self._worker_manager.relaunch_node(
                node, self._remove_exited_node
            )
        elif node.type == NodeType.PS:
            plan = self._ps_manager.relaunch_node(
                node, self._remove_exited_node
            )
        elif node.type == NodeType.EVALUATOR:
            plan = self._evaluator_manager.relaunch_node(
                node, self._remove_exited_node
            )
        elif node.type == NodeType.CHIEF or node.type == NodeType.MASTER:
            plan = self._chief_manager.relaunch_node(
                node, self._remove_exited_node
            )
        else:
            logger.error("Not support node type %s", node.type)
        self._set_ps_addrs_in_plan(plan)
        if self._remove_exited_node:
            plan.remove_nodes.append(node)
        node.relaunchable = False  # Avoid repeatedly relaunching the node.
        self._scaler.scale(plan)

    def clear_exited_nodes(self):
        if not self._remove_exited_node:
            return
        scale_plan = ScalePlan()
        with self._lock:
            for _, nodes in self._job_nodes.items():
                for _, node in nodes.items():
                    if not node.is_released and node.exited():
                        scale_plan.remove_nodes.append(node)
                        node.is_released = True
        if len(scale_plan.remove_nodes) > 0:
            logger.info(f"Remove exited nodes {scale_plan.remove_nodes}")
            self._scaler.scale(scale_plan)

    def clear_all_nodes(self):
        scale_plan = ScalePlan()
        with self._lock:
            for _, nodes in self._job_nodes.items():
                for _, node in nodes.items():
                    if not node.is_released:
                        scale_plan.remove_nodes.append(node)
                        node.is_released = True
        logger.info("Remove all nodes.")
        self._scaler.scale(scale_plan)

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
            logger.info("plan %s", plan.to_json())
            self._scaler.scale(plan)

    def get_running_nodes(self):
        nodes = self._chief_manager.get_running_nodes()
        nodes.extend(self._worker_manager.get_running_nodes())
        nodes.extend(self._evaluator_manager.get_running_nodes())
        nodes.extend(self._ps_manager.get_running_nodes())
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
        self._stopped = True

    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        if not self._job_nodes:
            logger.warning(
                "Skip updating for job_nodes hasn't been initialized."
            )
            return
        node = self._job_nodes[node_type][node_id]
        node.update_resource_usage(cpu, memory, gpu_stats)
        cpu_percent = node.used_resource.cpu / node.config_resource.cpu
        if cpu_percent < _dlrover_context.hang_cpu_usage_percentage:
            if node.start_hang_time == 0:
                now = datetime.now()
                node.start_hang_time = now.timestamp()
        else:
            if node.start_hang_time > 0:
                now = datetime.now()
            node.start_hang_time = 0

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
        """Start auto scaling nodes to improve the training throughput."""
        self._job_autoscaler.start_auto_scaling()

    def _set_ps_addrs_in_plan(self, plan: ScalePlan):
        ps_addrs = self._ps_manager.get_ps_addrs()
        plan.ps_addrs.extend(ps_addrs)

    def all_running_node_hanged(self):
        node_hang = self._worker_manager.running_nodes_hanged()
        node_hang.extend(self._chief_manager.running_nodes_hanged())
        node_hang.extend(self._ps_manager.running_nodes_hanged())
        node_hang.extend(self._evaluator_manager.running_nodes_hanged())
        if node_hang:
            return all(node_hang)
        return False

    def remove_not_joined_rdzv_workers(self, worker_ranks):
        plan = self._worker_manager.remove_not_joined_rdzv_workers(
            worker_ranks
        )
        self._scaler.scale(plan)

    def pend_without_workers(self):
        """Check whether to wait for evicted workers."""
        if self._worker_manager.has_exited_worker():
            return False
        elif self._worker_manager.wait_worker_restart():
            return True
        else:
            return False

    def handle_training_failure(
        self, node_type, node_id, restart_count=-1, error_data="", level=""
    ):
        """Process the training failure reported by the node."""
        node = self._job_nodes[node_type][node_id]
        if node.is_released:
            logger.info(f"The node {node.name} has been released.")
            return
        reluanch_node = self._error_monitor.process_error(
            node, restart_count, error_data, level
        )
        if reluanch_node and node.relaunchable:
            self._relaunch_node(node)

    def update_allreduce_node_unit(self, node_unit):
        if isinstance(self._job_optimizer, AllreduceJobResourceOptimizer):
            self._job_optimizer.set_node_unit(node_unit)

    def get_opt_strategy(self) -> ParallelConfig:
        strategy = self._job_strategy_generator.generate_opt_strategy()
        return strategy

    def update_node_paral_config(self, node_type, node_id, paral_config):
        node = self._job_nodes[node_type][node_id]
        node.update_paral_config(paral_config)

    def verify_restarting_worker_training(self, node_type, node_id):
        if node_type != NodeType.WORKER:
            return False
        return self._worker_manager.verify_restarting_training(node_id)

    def collect_node_heart_beat(self, node_type, node_id, timestamp):
        with self._lock:
            node = self._job_nodes[node_type][node_id]
            if node.heartbeat_time == 0:
                logger.info(
                    f"Start receiving heartbeat from node {node_id}"
                    f"-{node.name}"
                )
            node.heartbeat_time = timestamp

    def update_node_required_info_callback(self):
        self._worker_manager.update_node_required_info(self._nodes_required)


def create_job_manager(args: JobArgs, speed_monitor) -> DistributedJobManager:
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
    node_error_monitor = K8sJobErrorMonitor(
        args.namespace, args.cordon_fault_node
    )

    return DistributedJobManager(
        job_args=args,
        critical_worker_index=critical_worker_index,
        wait_pending_relaunch=wait_pending_relaunch,
        speed_monitor=speed_monitor,
        job=elastic_job,
        node_watcher=node_watcher,
        job_scaler=job_scaler,
        error_monitor=node_error_monitor,
    )
