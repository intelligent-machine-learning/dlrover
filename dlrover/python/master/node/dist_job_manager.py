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
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from dlrover.python.common.comm import ParallelConfig
from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    EventReportConstants,
    JobExitReason,
    JobStage,
    NodeEventType,
    NodeExitReason,
    NodeResourceLimit,
    NodeStatus,
    NodeType,
    TrainingExceptionLevel,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeEvent, NodeGroupResource
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
    NodeAction,
)
from dlrover.python.master.node.event_callback import (
    ClusterContext,
    NodeEventCallback,
)
from dlrover.python.master.node.job_auto_scaler import (
    JobAutoScaler,
    new_job_auto_scaler,
)
from dlrover.python.master.node.job_context import get_job_context
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
from dlrover.python.master.watcher.base_watcher import NodeWatcher
from dlrover.python.master.watcher.factory import (
    new_node_watcher,
    new_scale_plan_watcher,
)
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.scheduler.job import ElasticJob, JobArgs
from dlrover.python.training_event import DLRoverMasterEvent
from dlrover.python.util import k8s_util

_dlrover_context = Context.singleton_instance()
_master_evt = DLRoverMasterEvent().singleton_instance()
job_ctx = get_job_context()

_MAX_POD_RELAUNCH_COUNT = 5


def is_positive_exit(exit_reason):
    if exit_reason in [NodeExitReason.DIAG_FAIL, NodeExitReason.NO_HEARTBEAT]:
        return True
    return False


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
        perf_monitor=None,
        job=None,
        node_watcher: Optional[NodeWatcher] = None,
        job_scaler=None,
        external_config=None,
    ):
        super().__init__(
            job_args=job_args,
            perf_monitor=perf_monitor,
            external_config=external_config,
        )
        self._remove_exited_node = job_args.remove_exited_node
        node_restart_count: Dict[str, int] = {}
        for type, node_args in job_args.node_args.items():
            self._job_resource.node_group_resources[type] = (
                node_args.group_resource
            )
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
                "ParameterServerStrategy/AllReduceStrategy."
            )
        logger.info(f"New job optimizer: {self._job_optimizer.__class__}")

        worker_restart_count = node_restart_count.get(NodeType.WORKER, 0)
        ps_restart_count = node_restart_count.get(NodeType.PS, 0)

        self._relaunch_on_worker_failure = min(
            worker_restart_count, _MAX_POD_RELAUNCH_COUNT
        )
        _dlrover_context.max_relaunch_count = self._relaunch_on_worker_failure
        self._wait_pending_relaunch = wait_pending_relaunch
        self._start_launch_waiting_workers_time = time.time()
        self._critical_worker_index = critical_worker_index
        self._ps_relaunch_max_num = min(
            ps_restart_count, _MAX_POD_RELAUNCH_COUNT
        )
        logger.info(
            f"Worker relaunch number: {self._relaunch_on_worker_failure}; "
            f"PS relaunch number: {self._ps_relaunch_max_num}; "
            f"Critical worker index: {self._critical_worker_index}."
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
        self._relaunched_groups: List[int] = []
        self._group_relaunch_count = 0
        self._max_group_relaunch_count = _dlrover_context.max_relaunch_count

    def start(self):
        self._scaler.start()
        self._job_optimizer.update_job_uuid(self._job_args.job_uuid)
        self._job_optimizer.init_job_resource(self._job_resource)
        self._adjust_worker_for_estimator()
        self._init_nodes()
        self._init_job_auto_scaler()
        plan = self._create_initial_scale_plan()
        if not self._has_running_workers():
            # The job relaunches the evicted master, there are alive
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
        self._job_context.update_total_worker_num(worker_num)
        self._perf_monitor.set_target_worker_num(worker_num)
        self._training_node_config.set_node_num(worker_num)
        threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        ).start()
        threading.Thread(
            target=self._monitor_nodes_heartbeat,
            name="node_heartbeat_monitor",
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

    def get_ps_num(self):
        return self._job_resource.ps_num

    def get_job_type(self):
        return self._job_args.distribution_strategy

    def is_all_reduce_type_job(self):
        return (
            self._job_args.distribution_strategy
            == DistributionStrategy.ALLREDUCE
        )

    def restart(self):
        if not self.is_all_reduce_type_job():
            logger.warning(
                "Job restarting is only supported for ALLREDUCE type."
            )
            return

        restart_count = job_ctx.inc_job_restart_count()
        if restart_count > _dlrover_context.job_max_restart_count:
            logger.warning(
                f"Job restarting count over limit({restart_count}/{_dlrover_context.job_max_restart_count}), "
                f"request stop directly."
            )
            job_ctx.request_stop(1, JobExitReason.RESTART_OVER_LIMIT)
            return

        logger.info("Restarting job by relaunch all nodes.")
        target_nodes: List[Node] = []
        target_ranks = set([i for i in range(self.get_worker_num())])
        to_release_nodes: List[Node] = []
        for _, node in sorted(
            self._job_context.job_nodes_by_type(NodeType.WORKER).items(),
            key=lambda x: x[0],
            reverse=True,
        ):
            node_rank = node.rank_index
            if node_rank in target_ranks:
                logger.debug(
                    f"Add node {node.id} with rank {node.rank_index} for restarting relaunch."
                )
                target_nodes.append(node)
                target_ranks.remove(node_rank)
            else:
                if node.status in [
                    NodeStatus.PENDING,
                    NodeStatus.RUNNING,
                    NodeStatus.FINISHED,
                    NodeStatus.FAILED,
                ]:
                    logger.debug(
                        f"Add node {node.id} to release for restarting relaunch."
                    )
                    to_release_nodes.append(node)

        plan = self._worker_manager.relaunch_nodes(target_nodes, True)
        for to_release_node in to_release_nodes:
            plan.remove_nodes.append(to_release_node)

        self._scaler.scale(plan)

    def should_early_stop(self):
        # node-check all failed
        if (
            self.is_all_reduce_type_job()
            and self._worker_manager.is_all_initial_workers_node_check_failed(
                self.get_worker_num()
            )
        ):
            msg = (
                "Stop the training early because all worker nodes has "
                "failed the node check in rendezvous."
            )

            self._process_error(
                None,
                0,
                msg,
                level=TrainingExceptionLevel.RDZV_ERROR,
            )

            self._report_event(
                EventReportConstants.TYPE_INFO,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_EARLY_STOP,
                "All node check failed",
                {"nodes": json.dumps(self._worker_manager.cur_nodes)},
            )

            return True, JobExitReason.NODE_CHECK_FAILED, msg

        # ps pending judgement: any ps pod pending timeout
        timeout_ps_nodes = (
            self._ps_manager.get_pending_timeout_oom_recovered_node()
        )

        if len(timeout_ps_nodes) > 0:
            msg = (
                "Stop the training early because the nodes recovered from OOM "
                "are pending too long and have timed out."
            )

            self._process_error(
                timeout_ps_nodes[0],
                0,
                msg,
                level=TrainingExceptionLevel.ERROR,
            )
            self._report_event(
                EventReportConstants.TYPE_INFO,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_EARLY_STOP,
                "PS OOM",
                {},
            )
            return True, JobExitReason.PENDING_TIMEOUT, msg

        # ps/worker pending judgement:
        first_pending_node = (
            self._ps_manager.find_pending_node_caused_training_hang(
                self.get_ps_num(), self.get_job_type()
            )
            or self._worker_manager.find_pending_node_caused_training_hang(
                self.get_worker_num(), self.get_job_type()
            )
        )
        if first_pending_node is not None:
            msg = (
                "Stop the training early because 1) there is node pending "
                "2) alive nodes number consistently less than the min "
                "training nodes required 3) pending time last exceed limit."
            )
            self._process_error(
                None, 0, msg, level=TrainingExceptionLevel.ERROR
            )

            self._report_event(
                EventReportConstants.TYPE_INFO,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_EARLY_STOP,
                "Pending nodes",
                {
                    "first_pending_node": first_pending_node,
                },
            )
            return True, JobExitReason.PENDING_TIMEOUT, msg

        # insufficient worker judgement
        if (
            self.is_all_reduce_type_job()
            and self._worker_manager.is_training_hang_by_insufficient_worker()
        ):
            msg = (
                "Stop the training early because there isn't enough node to "
                "keep training."
            )

            self._process_error(
                None, 0, msg, level=TrainingExceptionLevel.ERROR
            )
            self._report_event(
                EventReportConstants.TYPE_INFO,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_EARLY_STOP,
                "Not enough nodes",
                {"nodes": json.dumps(self._worker_manager.cur_nodes)},
            )
            return True, JobExitReason.UNCOMPLETED_TIMEOUT, msg

        # no need to early stop
        return False, "", ""

    def handle_node_group_pending(self):
        pending_groups = self._worker_manager.get_pending_node_groups(
            self.get_job_type()
        )
        for node_group in pending_groups:
            if self._should_relaunch_node_group(node_group):
                self._relaunch_node_group(node_group)

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
            self._job_resource,
            self._ps_relaunch_max_num,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._chief_manager = ChiefManager(
            self._job_resource,
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._worker_manager = WorkerManager(
            self._job_resource,
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._evaluator_manager = EvaluatorManager(
            self._job_resource,
            self._relaunch_on_worker_failure,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

    def add_node_event_callback(self, node_event_callback):
        self._node_event_callbacks.append(node_event_callback)

    def _init_nodes(self):
        job_nodes = self._job_resource.init_job_node_meta(
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
            job_nodes,
            self._ps_is_critical,
            self._ps_relaunch_max_num,
            self._critical_worker_index,
        )
        update_nodes_priority(job_nodes)
        self._job_context.update_job_nodes(job_nodes)

        self._ps_manager.update_nodes_iter()
        self._chief_manager.update_nodes_iter()
        self._worker_manager.update_nodes_iter()
        self._evaluator_manager.update_nodes_iter()

    def _init_job_auto_scaler(self):
        self._job_autoscaler: JobAutoScaler = new_job_auto_scaler(
            self._job_args.distribution_strategy,
            self._job_resource,
            self._job_optimizer,
            self._perf_monitor,
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
                for event in self._node_watcher.watch():
                    self._process_event_safely(event)
            except Exception as e:
                logger.warning(e)
                time.sleep(30)
            time.sleep(5)

    def _monitor_node_heart_beat(self):
        with self._lock:
            try:
                events = self._get_dead_node_event()
            except Exception as e:
                logger.warning(e)
                events = []

        for event in events:
            self._process_event_safely(event)

    def _monitor_nodes_heartbeat(self):
        logger.info("Start node heartbeat monitoring.")
        while True:
            if self._stopped:
                logger.info("Stop node heartbeat monitoring.")
                break

            # deal with heartbeat
            self._monitor_node_heart_beat()

            time.sleep(15)

    def _get_dead_node_event(self, window_interval=600) -> List[NodeEvent]:
        now = time.time()
        dead_events: List[NodeEvent] = []
        job_nodes = self.get_job_nodes()
        logger.debug(f"Current job nodes are: {job_nodes}.")
        for nodes in job_nodes.values():
            for node in list(nodes.values()):
                if (
                    node.heartbeat_time > 0
                    and now - node.heartbeat_time > window_interval
                    and node.start_time
                    and node.create_time
                    and node.status == NodeStatus.RUNNING
                    and not node.is_succeeded_and_exited()
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
                    self._process_error(
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
                    self._event_reporter.report_node_no_heartbeat(
                        node, window_interval
                    )
        return dead_events

    def _get_nodes_time_info(self):
        result = {}
        job_nodes = self.get_job_nodes()
        for _, nodes in job_nodes.items():
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

    def _process_list_nodes(self, nodes: List[Node]):
        """Callback with node list by the list api of k8s."""

        logger.debug(f"Got list nodes: {nodes}")
        exist_nodes: Dict[str, List[Node]] = {}
        job_nodes = self.get_job_nodes()
        for node_type in job_nodes.keys():
            exist_nodes[node_type] = []

        if nodes:
            for node in nodes:
                node_type = node.type
                node_id = node.id
                exist_nodes[node_type].append(node)

                # for nodes not in current 'job_nodes' obj, re add it
                if (
                    node_id not in job_nodes[node_type]
                    and node.status != NodeStatus.DELETED
                ):
                    logger.info(
                        f"Node {node_type} {node.id} with status {node.status}"
                        " is re added without the event"
                    )
                    new_node = copy.deepcopy(node)
                    self._job_context.update_job_node(new_node)
                    if new_node.has_group():
                        self._job_context.update_job_node_by_group(new_node)

                # update node group info if necessary
                if (
                    node_type == NodeType.WORKER
                    and node_id in job_nodes[node_type]
                    and job_nodes[node_type][node_id].group != node.group
                ):
                    job_nodes[node_type][node_id].group = node.group
                    job_nodes[node_type][node_id].group_size = node.group_size
                    job_nodes[node_type][node_id].group_id = node.group_id

                if node.status == NodeStatus.DELETED:
                    event_type = NodeEventType.DELETED
                else:
                    event_type = NodeEventType.MODIFIED
                # Mock event to avoid missing events
                event = NodeEvent(event_type, node)
                self._process_event_safely(event)

        for node_type in job_nodes.keys():
            #  Avoid dictionary keys changed during iteration
            type_nodes = list(job_nodes[node_type].values())
            for node in type_nodes:
                if (
                    node.status != NodeStatus.INITIAL
                    and not node.is_released
                    and node.id
                    not in [node.id for node in exist_nodes[node_type]]
                ):
                    logger.info(
                        f"Node {node_type} {node.id} is deleted without the event"
                    )
                    new_node = copy.deepcopy(node)
                    new_node.is_released = True
                    new_node.status = NodeStatus.DELETED
                    event = NodeEvent(NodeEventType.DELETED, new_node)
                    self._process_event_safely(event)
                elif (
                    node.status == NodeStatus.INITIAL
                    and not node.is_released
                    and node.id
                    not in [node.id for node in exist_nodes[node_type]]
                ):
                    for exist_node in exist_nodes[node_type]:
                        if (
                            node.rank_index == exist_node.rank_index
                            and node.id < exist_node.id
                        ):
                            logger.info(
                                f"Node {node_type} {node.id} with "
                                f"rank {node.rank_index} is relaunched "
                                "by new node without the event"
                            )
                            new_node = copy.deepcopy(node)
                            new_node.is_released = True
                            new_node.status = NodeStatus.DELETED
                            new_node.exit_reason = NodeExitReason.RELAUNCHED
                            event = NodeEvent(NodeEventType.DELETED, new_node)
                            self._process_event_safely(event)

    def close_job(self):
        plan = ScalePlan()
        ps_resource = NodeGroupResource.new_empty()
        worker_resource = NodeGroupResource.new_empty()
        plan.node_group_resources = {
            "worker": worker_resource,
            "ps": ps_resource,
        }
        self._scaler.scale(plan=plan)
        os._exit(0)

    def _get_pod_unique_labels(self, node: Node):
        return {
            ElasticJobLabel.JOB_KEY: self._job_args.job_name,
            ElasticJobLabel.REPLICA_TYPE_KEY: node.type,
            ElasticJobLabel.RANK_INDEX_KEY: node.rank_index,
            ElasticJobLabel.REPLICA_INDEX_KEY: node.id,
        }

    def process_diagnosis_action(self, action: DiagnosisAction):
        if not action or isinstance(action, NoAction):
            return

        if isinstance(action, EventAction):
            self._report_event(
                action.event_type,
                action.event_instance,
                action.event_action,
                action.event_msg,
                action.event_labels,
            )
        elif isinstance(action, NodeAction):
            self._process_node_action(action)
        else:
            logger.info(f"Unsupported action for now: {action}")

    def _process_node_action(self, action: NodeAction):
        target_node = self._job_context.job_node(
            action.node_type, action.node_id
        )
        if not target_node:
            logger.warning(
                f"Node {target_node} was diagnosed as abnormal "
                "node, but not found in context."
            )
            return

        logger.info(
            f"Node {target_node} was diagnosed as abnormal node, "
            "trigger fault tolerance procedure."
        )

        event_node = copy.deepcopy(target_node)
        event_node.status = NodeStatus.FAILED
        event_node.exit_reason = NodeExitReason.DIAG_FAIL
        event = NodeEvent(
            event_type=NodeEventType.DELETED,
            node=event_node,
        )
        self._process_event_safely(event)

    def _process_event_safely(self, event: NodeEvent):
        try:
            self._process_event(event)
        except Exception as e:
            logger.warning(e)
            detail_trace_back = traceback.format_exc()
            logger.warning(detail_trace_back)

    def _process_event(self, event: NodeEvent):
        node_type = event.node.type
        node_status = event.node.status
        node_id = event.node.id
        node_name = event.node.name

        # Skip deleted event of pod if the cluster has relaunched a new pod
        # with the same type and rank as the deleted pod.
        if (
            event.event_type == NodeEventType.DELETED
            or node_status == NodeStatus.DELETED
        ) and not is_positive_exit(event.node.exit_reason):
            pod_labels_selector = k8s_util.gen_k8s_label_selector_from_dict(
                self._get_pod_unique_labels(event.node)
            )
            logger.debug(
                f"Recheck running pod with labels: {pod_labels_selector} "
                f"for deleted event."
            )
            pods = self._k8s_client.list_namespaced_pod(pod_labels_selector)
            if (
                pods
                and len(pods.items) > 0
                and any(
                    pod.status.phase == NodeStatus.RUNNING
                    and not pod.metadata.deletion_timestamp
                    for pod in pods.items
                )
            ):
                logger.info(
                    f"Skip deleted event for node: {node_id}({node_name}) "
                    "because same running pod already exists by "
                    f"labels: {pod_labels_selector} "
                )
                return

        job_nodes = self.get_job_nodes()
        if node_id not in job_nodes[node_type]:
            logger.info(f"The node {event.node.name} is released.")
            return
        else:
            cur_node = job_nodes[node_type][node_id]
            logger.debug(
                f"Update node({cur_node.id}), "
                f"name: {cur_node.name}->{event.node.name}, "
                f"status: {cur_node.status}->{event.node.status}, "
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
                is_released=event.node.is_released,
            )
            self._job_context.update_job_node(cur_node)
            if cur_node.has_group():
                self._job_context.update_job_node_by_group(cur_node)

        # For the given node id, check whether it meets
        # the state change condition
        if event.event_type == "exit":
            self.close_job()
            self._report_event(
                EventReportConstants.TYPE_INFO,
                self._job_args.job_name,
                EventReportConstants.ACTION_STOP,
                "",
                {},
            )
        new_status = event.node.status
        with self._lock:
            old_status = cur_node.status
            status_change_flow: NodeStateFlow = get_node_state_flow(
                old_status, event.event_type, new_status
            )
            # If there is no matched state change, return directly
            # If the node status is success, return directly
            if (
                status_change_flow is None
                or status_change_flow.from_status == NodeStatus.SUCCEEDED
            ):
                return

            # Update the node status
            cur_node.update_status(new_status)
            new_status = status_change_flow.to_status
            cur_node.set_exit_reason(event.node.exit_reason)
            self._job_context.update_job_node(cur_node)
            if cur_node.has_group():
                self._job_context.update_job_node_by_group(cur_node)

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
        self._event_reporter.report_node_status_change(
            cur_node, old_status, new_status
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

    def _should_relaunch_node_group(self, node_group: int) -> bool:
        """
        If node relaunch pending in node group happened, check if we
        need to relaunch the whole node group
        """
        logger.debug(
            f"Check node group {node_group} "
            f"already relaunched: {self._relaunched_groups}"
        )
        if node_group in self._relaunched_groups:
            logger.debug(
                f"Skip node group {node_group} due to "
                f"already relaunched: {self._relaunched_groups}"
            )
            return False

        node_check = all(
            node.relaunchable
            for node in job_ctx.job_node_group(node_group).values()
        )
        should_relaunch = (
            self._enable_relaunch_node
            and node_check
            and not job_ctx.is_stopping()
        )
        logger.info(
            f"Recheck node_group {node_group} can relaunch: {should_relaunch} with "
            f"{self._enable_relaunch_node}, {node_check}, {job_ctx.get_job_stage()}"
        )

        if self._group_relaunch_count > self._max_group_relaunch_count:
            logger.info(
                f"Node group {node_group} has exceeded max relaunch count: "
                f"{self._group_relaunch_count}/{self._max_group_relaunch_count}"
            )
            return False

        return should_relaunch

    def _should_relaunch(
        self, node: Node, status_change_flow: NodeStateFlow
    ) -> object:
        should_relaunch = (
            status_change_flow.should_relaunch
            and self._enable_relaunch_node
            and node.relaunchable
        )
        msg = ""
        if should_relaunch:
            logger.info(
                f"Recheck should_relaunch with {node}, "
                f"resource: {node.config_resource.to_resource_dict()}, "
                f"job_stage: {job_ctx.get_job_stage()}"
            )
            if job_ctx.is_stopping():
                should_relaunch = False
                msg = "Disable relaunch when job is stopping"
                logger.warning(
                    f"Disable {node.name}/{node.id} relaunch when job is stopping."
                )
            elif job_ctx.is_restarting():
                should_relaunch = False
                msg = "Disable relaunch when job is restarting"
                logger.warning(
                    f"Disable {node.name}/{node.id} relaunch when job is restarting."
                )
            elif (
                node.exit_reason == NodeExitReason.FATAL_ERROR
                and not _dlrover_context.relaunch_always
            ):
                should_relaunch = False
                msg = "Disable relaunch due to fatal error"
            elif node.exit_reason == NodeExitReason.RELAUNCHED:
                should_relaunch = False
                msg = "Disable relaunch due to already relaunched"
            elif node.exit_reason == NodeExitReason.OOM:
                mem = node.config_resource.memory
                if self.is_all_reduce_type_job():
                    should_relaunch = False
                    logger.warning(
                        "The all-reduce type job will not try to recover node "
                        f"error with oom issue, node: {node.name}."
                    )
                elif mem >= NodeResourceLimit.MAX_MEMORY:
                    should_relaunch = False
                    logger.warning(
                        f"The memory of node {mem} is beyond the limit "
                        f"{NodeResourceLimit.MAX_MEMORY} MB, "
                        f"node: {node.name}."
                    )
                    msg = f"{mem} beyond {NodeResourceLimit.MAX_MEMORY}"
                elif node.relaunch_count >= node.max_relaunch_count:
                    should_relaunch = False
                    logger.warning(
                        f"The relaunched count {node.relaunch_count} is "
                        f"beyond the maximum {node.max_relaunch_count} "
                        f"for node: {node.name}."
                    )
                    msg = (
                        f"Relaunched {node.relaunch_count} "
                        f"beyond {node.max_relaunch_count}"
                    )
                else:
                    node.is_recovered_oom = True
                    self._job_optimizer.adjust_oom_resource(node)
            elif node.exit_reason != NodeExitReason.KILLED:
                if node.relaunch_count >= node.max_relaunch_count:
                    logger.warning(
                        "The relaunch count "
                        f"{node.relaunch_count}/{node.max_relaunch_count} "
                        f"has been exhausted for node: {node.name}."
                    )
                    should_relaunch = False
                    msg = f"{node.relaunch_count} exhausted {node.max_relaunch_count}"

        if should_relaunch:
            node.relaunch_count += 1
            logger.info(
                f"Node {node.name} passed should_relaunch with "
                f"{node.relaunch_count}/{node.max_relaunch_count}"
            )

        if not should_relaunch and len(msg) > 0:
            self._report_event(
                EventReportConstants.TYPE_INFO,
                node.get_name(),
                EventReportConstants.ACTION_NOT_RELAUNCH,
                msg,
                {},
            )

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
            return

        if plan and len(plan.launch_nodes) > 0:
            self._event_reporter.report_node_relaunch(
                node, plan.launch_nodes[0]
            )

        self._set_ps_addrs_in_plan(plan)
        if self._remove_exited_node:
            plan.remove_nodes.append(node)

        # Avoid repeatedly relaunching the node.
        node.relaunchable = False

        self._job_context.update_job_node(node)
        if node.has_group():
            self._job_context.update_job_node_by_group(node)
        self._scaler.scale(plan, with_merge=True)

    def _relaunch_node_group(self, node_group: int):
        """
        Relaunch all nodes in the node_group group, no matter it is
        running or pending.
        The group index should be different with all existing group,
        so we use max_group_idx to keep it in record

        """
        group_idx = self._job_context.next_group_idx()
        logger.info(
            f"Relaunch node group {node_group} with group_idx {group_idx}"
        )

        launch_nodes: List[Node] = []
        group_nodes = list(
            self._job_context.job_node_group(node_group).values()
        )
        for node in group_nodes:
            if node.type != NodeType.WORKER:
                logger.warning(f"{node.name} is not worker node")
                continue

            # update node.group to max_group_idx
            old_node = copy.deepcopy(node)
            old_node.group = group_idx
            old_node.group_id = ""
            launch_nodes.append(old_node)

        plan = self._worker_manager.relaunch_nodes(launch_nodes, True)

        for node in group_nodes:
            plan.remove_nodes.append(node)
            node.relaunchable = False
            self._job_context.update_job_node(node)
            if node.has_group():
                self._job_context.update_job_node_by_group(node)

        logger.info(
            f"Finish scale plan for node group {node_group} relaunch "
            f"group_nodes: {group_nodes} "
            f"launch_nodes: {plan.launch_nodes} "
            f"remove_nodes: {plan.remove_nodes} "
        )

        self._relaunched_groups.append(node_group)
        self._scaler.scale(plan)
        self._group_relaunch_count += 1
        return plan

    def clear_exited_nodes(self):
        if not self._remove_exited_node:
            return
        job_nodes = self.get_job_nodes()
        scale_plan = ScalePlan()
        with self._lock:
            for _, nodes in job_nodes.items():
                for _, node in nodes.items():
                    if not node.is_released and node.exited():
                        scale_plan.remove_nodes.append(node)
                        node.is_released = True
                        self._job_context.update_job_node(node)
                        if node.has_group():
                            self._job_context.update_job_node_by_group(node)
        if len(scale_plan.remove_nodes) > 0:
            logger.info(f"Remove exited nodes {scale_plan.remove_nodes}")
            self._scaler.scale(scale_plan)

    def clear_all_nodes(self):
        scale_plan = ScalePlan()
        job_nodes = self.get_job_nodes()
        with self._lock:
            for _, nodes in job_nodes.items():
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
        job_nodes = self.get_job_nodes()
        for _, nodes in job_nodes.items():
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
        job_nodes = self.get_job_nodes()
        if job_nodes[NodeType.WORKER][worker_id].critical:
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
        if not self._stopped:
            self._enable_relaunch_node = False
            job_nodes = self.get_job_nodes()
            with self._lock:
                for node_type in job_nodes.keys():
                    for node in job_nodes[node_type].values():
                        node.critical = False
                        node.is_released = True
                        node.relaunchable = False
                        self._job_context.update_job_node(node)
                        if node.has_group():
                            self._job_context.update_job_node_by_group(node)
                for node in job_nodes[NodeType.WORKER].values():
                    node.eval_time = self._perf_monitor.get_worker_eval_time(
                        node.id
                    )
                    self._job_context.update_job_node(node)
                    if node.has_group():
                        self._job_context.update_job_node_by_group(node)
            self._stopped = True

    def update_node_resource_usage(
        self, node_type, node_id, cpu, memory, gpu_stats=[]
    ):
        node = self._job_context.job_node(node_type, node_id)
        if node is None:
            logger.warning(
                f"Skip update node[{node_type}][{node_id}] resources"
            )
            return
        node.update_resource_usage(cpu, memory, gpu_stats)
        if node.config_resource.cpu:
            if node.config_resource.gpu_num:
                # skip cpu hang for gpu case for now
                logger.debug("CPU hang calculation is skipped when gpu > 0.")
            else:
                cpu_percent = node.used_resource.cpu / node.config_resource.cpu
                if cpu_percent < _dlrover_context.hang_cpu_usage_percentage:
                    if node.start_hang_time == 0:
                        now = datetime.now()
                        node.start_hang_time = now.timestamp()
                else:
                    if node.start_hang_time > 0:
                        now = datetime.now()
                    node.start_hang_time = 0
                self._job_context.update_job_node(node)
        else:
            logger.warning(
                "CPU requests not configure "
                "and can not determine if the job node is hung"
            )

    def update_node_service_addr(self, node_type, node_id, service_addr):
        node = self._job_context.job_node(node_type, node_id)
        if node is None:
            logger.error(f"no Node[{node_type}][{node_id}] found")
            return
        node.update_service_address(service_addr)
        node.status = NodeStatus.RUNNING
        node.is_released = False
        self._job_context.update_job_node(node)

    def get_cur_cluster_ps(self):
        """Get PS nodes in the current training cluster."""
        logger.info("job nodes are {}".format(self.get_job_nodes()))
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
        job_nodes = self.get_job_nodes()
        training_nodes = list(job_nodes[NodeType.WORKER].values()) + list(
            job_nodes[NodeType.PS].values()
        )
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
                self._job_context.update_job_node(node)
                plan.remove_nodes.append(node)
        self._scaler.scale(plan)

    def start_auto_scaling(self):
        """Start auto scaling nodes to improve the training throughput."""
        self._job_autoscaler.start_auto_scaling()

    def _set_ps_addrs_in_plan(self, plan: ScalePlan):
        ps_addrs = self._ps_manager.get_ps_addrs()
        plan.ps_addrs.extend(ps_addrs)

    def _report_event(
        self,
        event_type: str,
        instance: str,
        action: str,
        msg: str,
        labels: Dict[str, str],
    ):
        self._event_reporter.report(event_type, instance, action, msg, labels)

    def _process_error(
        self,
        node: Optional[Node],
        restart_count: int,
        error_data: str,
        level: str,
    ) -> bool:
        if node:
            if level == TrainingExceptionLevel.NODE_ERROR:
                self._job_context.report_failed_node(node.id)
            return self.process_error(node, restart_count, error_data, level)
        return False

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
        node = self._job_context.job_node(node_type, node_id)

        if error_data:
            # self detected reason override the reason from k8s pod
            node.set_exit_reason(error_data)
        else:
            # inherit the reason from k8s pod
            error_data = node.exit_reason
        logger.info(f"Handle failed node: {node} with reason: {error_data}")
        if node.is_released:
            return
        relaunch_node = self._process_error(
            node, restart_count, error_data, level
        )
        if relaunch_node and node.relaunchable:
            self._relaunch_node(node)

    def update_allreduce_node_unit(self, node_unit):
        if isinstance(self._job_optimizer, AllreduceJobResourceOptimizer):
            self._job_optimizer.set_node_unit(node_unit)

    def get_opt_strategy(self) -> ParallelConfig:
        strategy = self._job_strategy_generator.generate_opt_strategy()
        return strategy

    def update_node_paral_config(self, node_type, node_id, paral_config):
        node = self._job_context.job_node(node_type, node_id)
        if node is None:
            logger.warning(f"not found Node[{node_type}][{node_id}]")
            return
        node.update_paral_config(paral_config)
        self._job_context.update_job_node(node)

    def verify_restarting_worker_training(self, node_type, node_id):
        if node_type != NodeType.WORKER:
            return False
        return self._worker_manager.verify_restarting_training(node_id)

    def collect_node_heart_beat(
        self, node_type, node_id, timestamp
    ) -> DiagnosisAction:
        with self._lock:
            node = self._job_context.job_node(node_type, node_id)
            if node is None:
                return NoAction()
            if node.heartbeat_time == 0:
                logger.info(f"Start receiving heartbeat from node {node_id}")
            node.heartbeat_time = timestamp
            self._job_context.update_job_node(node)
            action = self._job_context.next_action(instance=node_id)
            if not action or isinstance(action, NoAction):
                return self._job_context.next_action(
                    instance=DiagnosisConstant.ANY_INSTANCE
                )
            else:
                logger.debug(f"Collect action from {node_id}: {action}")
                return action

    def update_node_required_info_callback(self):
        self._worker_manager.update_node_required_info(self._nodes_required)

    def process_reported_node_event(self, node_event: NodeEvent):
        """
        The node events here is reported from training agent.

        Args:
            node_event: The event from training agent.
        """

        event_type = node_event.event_type
        node = node_event.node
        node_type = node.type
        node_id = node.id

        with self._lock:
            target_node = self._job_context.job_node(node_type, node_id)
            if target_node:
                logger.info(
                    f"Node {node_id}({node_type}) reported status to {event_type}."
                )
                target_node.update_reported_status(event_type)
                self._job_context.update_job_node(target_node)

            if event_type == NodeEventType.SUCCEEDED_EXITED:
                self._job_context.update_job_stage(JobStage.JOB_STOPPING)
                logger.info(
                    f"Update job stage to {self._job_context.get_job_stage()} "
                    f"due to event {event_type}."
                )

    def get_node_required_info(self):
        return self._nodes_required

    def get_job_strategy(self):
        return self._job_args.distribution_strategy

    def _handle_node_error(self, node: Node, error_data: str):
        logger.info(
            f"{node.name} on {node.host_name} is down. Reason: {error_data}"
        )
        if self._job_args.cordon_fault_node:
            succeed = self._k8s_client.cordon_node(node.host_name)
            if succeed:
                logger.info(f"Host {node.host_name} is marked unscheduled.")
        return True


def create_job_manager(args: JobArgs, perf_monitor) -> DistributedJobManager:
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

    return DistributedJobManager(
        job_args=args,
        critical_worker_index=critical_worker_index,
        wait_pending_relaunch=wait_pending_relaunch,
        perf_monitor=perf_monitor,
        job=elastic_job,
        node_watcher=node_watcher,
        job_scaler=job_scaler,
    )
