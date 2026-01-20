# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Copyright 2026 The DLRover Authors. All rights reserved.
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

import importlib
import json
import threading
import time
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Dict, List, Optional

import grpc
from grpc import ServicerContext

from dlrover.python.common import comm
from dlrover.python.common.comm import BaseRequest, BaseResponse, TaskType
from dlrover.python.common.constants import (
    GRPC,
    CustomMetricKeys,
    JobConstant,
    JobStage,
    KeyValueOps,
    NodeEventType,
    NodeType,
    RendezvousName,
    TrainingExceptionLevel,
    TrainingLoopStatus,
    CommunicationReqType,
    CommunicationReqMeta,
    CommunicationType,
)
from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.event.train_event import TrainEventName
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeEvent
from dlrover.python.diagnosis.common.diagnosis_action import (
    NoAction,
    JobRestartAction,
    JobAbortionAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.kv_store_service import (
    KVStoreService,
)
from dlrover.python.master.elastic_training.rdzv_manager import (
    NetworkCheckRendezvousManager,
    RendezvousManager,
)
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.training_node import SyncNodeTrainingPorts
from dlrover.python.master.shard.dataset_splitter import new_dataset_splitter
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.watcher.base_watcher import Node
from dlrover.python.util.queue.queue import RayEventQueue

try:
    from dlrover.python.master.elastic_training.elastic_ps import (
        ElasticPsService,
    )
    from dlrover.python.master.elastic_training.sync_service import SyncService
except ImportError:
    logger.info("Run the master locally.")
    pass


_dlrover_context = Context.singleton_instance()
_DEFAULT_NUM_MINIBATCHES_PER_SHARD = 100
ray_event_queue = RayEventQueue.singleton_instance()
_event_context = JobEventContext.singleton_instance()
_job_ctx = get_job_context()


class MasterServicer(ABC):
    """Master service base class."""

    def __init__(
        self,
        task_manager,
        job_manager,
        perf_monitor: PerfMonitor,
        rdzv_managers: Dict[str, RendezvousManager],
        diagnosis_manager: DiagnosisMaster,
        job_metric_collector=None,
        elastic_ps_service=None,
        sync_service=None,
    ):
        self._task_manager: TaskManager = task_manager
        self._job_manager = job_manager
        self._perf_monitor = perf_monitor
        self._rdzv_managers = rdzv_managers
        self._diagnosis_manager = diagnosis_manager
        self._kv_store = KVStoreService()
        self._job_metric_collector: JobMetricCollector = job_metric_collector
        self._elastic_ps_service: ElasticPsService = elastic_ps_service
        self._sync_service: SyncService = sync_service
        self._lock = threading.Lock()
        self._version = 0
        self._start_training_time = 0
        self._start_autoscale = False
        self._event_reporter = get_event_reporter()

        # preload module for class reflection
        self._diagnosis_data_module = importlib.import_module(
            "dlrover.python.diagnosis.common.diagnosis_data"
        )
        # clear kv store in case previous data is still there
        self._kv_store.clear()

    @abstractmethod
    def get_response(self, method):
        """Should be implemented by subclasses."""
        pass

    @abstractmethod
    def get_task_type(self, task_type):
        """Should be implemented by subclasses."""
        pass

    def validate_request(self, request_meta: dict) -> bool:
        if CommunicationReqMeta.COMM_META_JOB_UID in request_meta:
            req_job_uid = request_meta[CommunicationReqMeta.COMM_META_JOB_UID]
            # 1) for backward compatible: skip if client uid is empty
            # 2) for local mode compatible: skip if uid is empty(local mode)
            if (
                req_job_uid
                and self._job_manager.job_uid
                and req_job_uid != self._job_manager.job_uid
            ):
                logger.error(
                    f"Invalid job uid: {req_job_uid} for request, "
                    f"expect: {self._job_manager.job_uid}."
                )
                return False
        return True

    def get(self, request, _):
        node_type = request.node_type
        node_id = request.node_id
        req_message = comm.deserialize_message(request.data)

        response = self.get_response(CommunicationReqType.COMM_REQ_TYPE_GET)
        if not req_message:
            return response
        message = None
        if isinstance(req_message, comm.TaskRequest):
            message = self._get_task(node_type, node_id, req_message)
        elif isinstance(req_message, comm.ShardCheckpointRequest):
            message = self._get_shard_checkpoint(req_message)
        elif isinstance(req_message, comm.ClusterVersionRequest):
            message = self._get_cluster_version(req_message)
        elif isinstance(req_message, comm.RunningNodesRequest):
            message = self._get_running_nodes()
        elif isinstance(req_message, comm.JoinRendezvousRequest):
            message = self._join_rendezvous(req_message)
        elif isinstance(req_message, comm.WaitingNodeNumRequest):
            message = self._num_nodes_waiting(req_message.rdzv_name)
        elif isinstance(req_message, comm.NetworkReadyRequest):
            message = self._check_fault_node()
        elif isinstance(req_message, comm.StragglerExistRequest):
            message = self._check_straggler()
        elif isinstance(req_message, comm.CommWorldRequest):
            message = self._get_comm_world(req_message)
        elif isinstance(req_message, comm.KeyValuePair):
            if req_message.op == KeyValueOps.ADD:
                message = self._kv_store_add(req_message)
            else:
                message = self._kv_store_get(req_message)
        elif isinstance(req_message, comm.KeyValuePairs):
            message = self._kv_store_multi_get(req_message)
        elif isinstance(req_message, comm.PsNodesRequest):
            message = self._query_ps_nodes()
        elif isinstance(req_message, comm.TrainingStatusRequest):
            message = self._get_training_status()
        elif isinstance(req_message, comm.ParallelConfigRequest):
            message = self._get_paral_config()
        elif isinstance(req_message, comm.CheckHardwareResetRequest):
            message = self._need_to_restart_training(node_type, node_id)
        elif isinstance(req_message, comm.SyncTrainingPort):
            message = self._sync_training_ports(node_id, req_message)
        elif isinstance(req_message, comm.ElasticRunConfigRequest):
            configs = self._job_manager.get_elastic_run_configs()
            message = comm.ElasticRunConfig(configs=configs)
        elif isinstance(req_message, comm.PreCheckRequest):
            message = self._get_pre_check_result(
                node_type, node_id, req_message
            )
        elif isinstance(req_message, comm.HeartBeat):
            message = self._report_heartbeat(node_type, node_id, req_message)

        if message:
            response.data = message.serialize()
        return response

    def _get_task(self, node_type, node_id, request: comm.TaskRequest):
        if not self._start_training_time:
            self._start_training_time = int(time.time())
        shard = comm.Shard()
        res = comm.Task(shard=shard)
        ds_name = request.dataset_name
        dataset = self._task_manager.get_dataset(ds_name)
        if not dataset:
            return res
        task = self._task_manager.get_dataset_task(node_type, node_id, ds_name)

        if task:
            res.task_id = task.task_id
            res.type = task.task_type
            res.shard.name = task.shard.name
            res.shard.start = task.shard.start
            res.shard.end = task.shard.end
            if task.shard.record_indices:
                res.shard.indices = task.shard.record_indices
        elif not dataset.completed():
            res.type = self.get_task_type(TaskType.WAIT)
        with self._lock:
            self._task_manager.reset_worker_start_task_time(node_id)
        return res

    def _get_shard_checkpoint(self, request: comm.ShardCheckpointRequest):
        response = comm.ShardCheckpoint()
        dataset = self._task_manager.get_dataset(request.dataset_name)
        checkpoint = dataset.checkpoint()
        if checkpoint:
            response.content = checkpoint.to_json()
        return response

    def _get_cluster_version(self, request: comm.ClusterVersionRequest):
        message = comm.ClusterVersion()
        if not self._elastic_ps_service:
            return message

        if request.task_type == NodeType.WORKER:
            message.version = self._elastic_ps_service.get_worker_version(
                request.version_type, request.task_id
            )
        elif request.task_type == NodeType.PS:
            message.version = self._elastic_ps_service.get_ps_version(
                request.version_type, request.task_id
            )
        return message

    def _query_ps_nodes(self):
        res = comm.PsNodes(nodes=[])
        training_ps: List[Node] = self._job_manager.get_next_cluster_ps()
        ready = self._job_manager.ready_for_new_ps_cluster()
        ps_failure = self._job_manager.has_ps_failure()
        for ps in training_ps:
            ps_meta = comm.NodeMeta()
            ps_meta.type = NodeType.PS
            ps_meta.addr = ps.service_addr
            ps_meta.cpu = ps.config_resource.cpu
            ps_meta.memory = int(ps.config_resource.memory)
            res.nodes.append(ps_meta)
        res.new_ps_ready = ready
        res.ps_failure = ps_failure
        return res

    def _get_running_nodes(self):
        res = comm.RunningNodes(nodes=[])
        nodes: List[Node] = self._job_manager.get_running_nodes()
        for node in nodes:
            meta = comm.NodeMeta()
            meta.type = node.type
            meta.addr = node.service_addr
            meta.cpu = node.config_resource.cpu
            meta.memory = node.config_resource.memory
            if node.config_resource.gpu_type:
                meta.gpu_type = node.config_resource.gpu_type
                meta.gpu = node.config_resource.gpu_num
            res.nodes.append(meta)
        return res

    def _get_training_status(self):
        res = comm.TrainingStatus()
        if self._task_manager.training_started():
            res.status = TrainingLoopStatus.START
        else:
            res.status = TrainingLoopStatus.PENDING
        return res

    def _check_fault_node(self):
        rdzv_manager: NetworkCheckRendezvousManager = self._rdzv_managers[
            RendezvousName.NETWORK_CHECK
        ]
        nodes, reason = rdzv_manager.check_fault_node()
        res = comm.NetworkCheckResult(nodes=nodes, reason=reason)
        return res

    def _check_straggler(self):
        rdzv_manager: NetworkCheckRendezvousManager = self._rdzv_managers[
            RendezvousName.NETWORK_CHECK
        ]
        nodes, reason = rdzv_manager.get_straggler()
        res = comm.NetworkCheckResult(nodes=nodes, reason=reason)
        return res

    def _join_rendezvous(self, request: comm.JoinRendezvousRequest):
        rdzv_manager = self._rdzv_managers[request.rdzv_name]
        node_rank = request.node_rank
        if node_rank == -1:  # Back compatibility
            node_rank = request.node_id
        round = rdzv_manager.join_rendezvous(
            request.node_id,
            node_rank,
            request.local_world_size,
            request.node_ip,
        )
        if request.rdzv_name == RendezvousName.NETWORK_CHECK:
            # The waiting node in the training rdzv should clear if
            # a worker join network-check rdzv.
            training_manager = self._rdzv_managers[RendezvousName.TRAINING]
            training_manager.clear_waiting_nodes()

        # Pause hang diagnosis during rendezvous
        if node_rank == 0:
            self._diagnosis_manager.pause_observing()

        res = comm.RendezvousState(round=round)
        return res

    def _num_nodes_waiting(self, rdzv_name):
        """
        Return the number of waiting nodes for a rendezvous.

        Args:
            rdzv_name: NodeCheck or ElasticTraining

        Returns:
            >0: the number of waiting nodes
            0: exception happened
            -1: the job is stopping, no more rendezvous

        """
        waiting_num = self._rdzv_managers[rdzv_name].num_nodes_waiting()
        if _job_ctx.get_job_stage() == JobStage.JOB_STOPPING:
            logger.debug(
                f"Job is stopping, set waiting_num {waiting_num} to -1"
            )
            waiting_num = -1
        res = comm.RendezvousState(waiting_num=waiting_num)
        return res

    def _get_comm_world(self, request: comm.CommWorldRequest):
        rdzv_manager = self._rdzv_managers[request.rdzv_name]
        rdzv_round, group, nodes = rdzv_manager.get_comm_world(request.node_id)
        res = comm.RendezvousState(world={})
        res.group = group
        res.round = rdzv_round
        for rank, meta in nodes.items():
            res.world[rank] = meta.process_num
        if nodes and request.rdzv_name == RendezvousName.TRAINING:
            rdzv_round = rdzv_manager.get_rdzv_round()
            metrics = {CustomMetricKeys.RDZV_ROUND: rdzv_round}
            if self._job_metric_collector:
                self._job_metric_collector.collect_custom_data(metrics)
            # Finish elastic training rendezvous so we continue diagnosis
            self._diagnosis_manager.continue_observing()
        logger.debug(f"_get_comm_world: {request} {res}")
        return res

    def _kv_store_get(self, request: comm.KeyValuePair):
        value = self._kv_store.get(request.key)
        res = comm.KeyValuePair(request.key, value)
        logger.debug(f"_kv_store_get: {request} {res}")
        return res

    def _kv_store_add(self, request: comm.KeyValuePair):
        value = self._kv_store.add(request.key, request.value)
        res = comm.KeyValuePair(request.key, value)
        logger.debug(f"_kv_store_add: {request} {res}")
        return res

    def _kv_store_multi_get(self, request: comm.KeyValuePairs):
        kvs: Dict[str, bytes] = {}
        for key in request.kvs.keys():
            value = self._kv_store.get(key)
            if value == b"":
                kvs = {}
                break
            else:
                kvs[key] = value

        res = comm.KeyValuePairs(kvs)
        logger.debug(f"_kv_store_multi_get: {request} {res}")
        return res

    def _get_paral_config(self):
        res = self._job_manager.get_opt_strategy()
        if not res:
            res = comm.ParallelConfig()
        return res

    def _need_to_restart_training(self, node_type, node_id):
        restart = self._job_manager.verify_restarting_worker_training(
            node_type, node_id
        )
        res = comm.ParallelConfig()
        res.restart = restart
        return res

    def report(self, request, _):
        node_type = request.node_type
        node_id = request.node_id
        message = comm.deserialize_message(request.data)

        response = self.get_response(CommunicationReqType.COMM_REQ_TYPE_REPORT)
        if not message:
            return response

        success = False
        if isinstance(message, comm.DatasetShardParams):
            success = self._collect_dataset_shard_params(message)
        elif isinstance(message, comm.ResourceStats):
            success = self._update_node_resource_usage(
                node_type, node_id, message
            )
        elif isinstance(message, comm.ModelInfo):
            success = self._collect_model_info(message)
        elif isinstance(message, comm.GlobalStep):
            success = self._collect_global_step(message)
        elif isinstance(message, comm.ShardCheckpoint):
            success = self._restore_shard_checkpoint(message)
        elif isinstance(message, comm.TaskResult):
            success = self._report_task_result(message)
        elif isinstance(message, comm.ClusterVersion):
            success = self._update_cluster_version(message)
        elif isinstance(message, comm.NodeAddress):
            success = self._update_node_address(message)
        elif isinstance(message, comm.NodeEvent):
            success = self._deal_with_reported_node_event(message)
        elif isinstance(message, comm.AtorchEvent):
            success = self._handle_reported_atorch_event(message)
        elif isinstance(message, comm.SyncJoin):
            success = self._join_sync(node_type, node_id, message)
        elif isinstance(message, comm.SyncFinish):
            success = self._sync_finished(message)
        elif isinstance(message, comm.SyncBarrier):
            success = self._barrier(message)
        elif isinstance(message, comm.NodeFailure):
            success = self._report_failure(node_type, node_id, message)
        elif isinstance(message, comm.RendezvousParams):
            success = self._report_rdzv_params(message)
        elif isinstance(message, comm.PsReady):
            success = self._ready_for_ps_relaunch()
        elif isinstance(message, comm.KeyValuePair):
            success = self._kv_store_set(message)
        elif isinstance(message, comm.KeyValuePairs):
            success = self._kv_store_multi_set(message)
        elif isinstance(message, comm.ParallelConfig):
            success = self._report_paral_config(node_type, node_id, message)
        elif isinstance(message, comm.NodeCheckpointState):
            success = self._sync_checkpoint(node_type, node_id, message)
        elif isinstance(message, comm.DiagnosisReportData):
            success = self._report_node_diagnosis_data(message)
        elif isinstance(message, comm.Event):
            success = self._report_event(message)
        elif isinstance(message, comm.RdzvBlocked):
            success = self.set_rdzv_blocked(message)
        elif isinstance(message, comm.DiagnosisAction):
            success = self._report_action(message)

        response.success = success
        return response

    def set_rdzv_blocked(self, message: comm.RdzvBlocked):
        rdzv_manager = self._rdzv_managers[RendezvousName.TRAINING]
        rdzv_manager.set_rdzv_blocked(message.blocked, message.reason)
        return True

    def _ready_for_ps_relaunch(self):
        self._job_manager.post_ps_ready()
        return True

    def _collect_dataset_shard_params(self, metrics: comm.DatasetShardParams):
        num_minibatches_per_task = (
            metrics.num_minibatches_per_shard
            or _DEFAULT_NUM_MINIBATCHES_PER_SHARD
        )
        shard_size = metrics.batch_size * num_minibatches_per_task
        splitter = new_dataset_splitter(
            metrics.shuffle,
            shard_size,
            metrics.dataset_size,
            metrics.num_epochs,
            metrics.dataset_name,
            metrics.storage_type,
        )
        self._task_manager.new_dataset(
            metrics.batch_size,
            metrics.dataset_size,
            metrics.dataset_name,
            splitter,
            metrics.task_type,
        )
        if self._job_metric_collector:
            self._job_metric_collector.collect_dataset_metric(
                metrics.dataset_name,
                metrics.dataset_size,
                metrics.storage_type,
            )
            if metrics.task_type == self.get_task_type(TaskType.TRAINING):
                self._job_metric_collector.collect_training_hyper_params(
                    metrics.num_epochs, metrics.batch_size
                )
        return True

    def _update_node_resource_usage(
        self, node_type, node_id, metrics: comm.ResourceStats
    ):
        logger.debug(
            f"Update resource usage for {node_type}-{node_id},"
            f"cpu={metrics.cpu}, memory={metrics.memory},"
            f"gpu_stats={metrics.gpu_stats}"
        )
        if self._job_manager:
            self._job_manager.update_node_resource_usage(
                node_type,
                node_id,
                metrics.cpu,
                metrics.memory,
                metrics.gpu_stats,
            )
        return True

    def _collect_model_info(self, metrics: comm.ModelInfo):
        if self._job_metric_collector:
            self._job_metric_collector.collect_model_metric(metrics)
        return True

    def _collect_global_step(self, metrics: comm.GlobalStep):
        self._perf_monitor.collect_global_step(metrics.step, metrics.timestamp)
        self._collect_runtime_stats()
        self._check_start_auto_scale_worker()
        return True

    def _restore_shard_checkpoint(self, message: comm.ShardCheckpoint):
        success = self._task_manager.restore_dataset_from_checkpoint(
            message.content
        )
        return success

    def _collect_runtime_stats(self):
        if self._job_metric_collector and self._job_manager:
            nodes = self._job_manager.get_running_nodes()
            self._job_metric_collector.collect_runtime_stats(
                self._perf_monitor, nodes
            )

    def _report_task_result(self, request: comm.TaskResult):
        success = True
        if request.err_message:
            logger.warning("Worker reported error: " + request.err_message)
            success = False
        task, _ = self._task_manager.report_dataset_task(request, success)
        if (
            not self._start_autoscale
            and self._job_manager
            and self._perf_monitor.completed_global_step == 0
            and int(time.time()) - self._start_training_time
            > _dlrover_context.seconds_to_autoscale_worker
        ):
            logger.info("Start autoscale for non-training jobs")
            self._job_manager.start_auto_scaling()
            self._start_autoscale = True

        if (
            self._job_metric_collector
            and task
            and task.task_type == self.get_task_type(TaskType.PREDICTION)
        ):
            self._collect_runtime_stats()
            self._check_start_auto_scale_worker()
        return success

    def _check_start_auto_scale_worker(self):
        sample_count = self._perf_monitor.get_sample_count()
        if (
            not self._start_autoscale
            and sample_count >= _dlrover_context.sample_count_to_adjust_worker
        ):
            logger.info(
                "Start autoscale with %s stats samples",
                sample_count,
            )
            self._job_manager.start_auto_scaling()
            self._start_autoscale = True

    def _update_cluster_version(self, message: comm.ClusterVersion):
        if not self._elastic_ps_service:
            return False

        if message.task_type == NodeType.WORKER:
            self._elastic_ps_service.update_worker_version(
                message.task_id, message.version_type, message.version
            )
        elif message.task_type == NodeType.PS:
            self._elastic_ps_service.update_ps_version(
                message.task_id, message.version_type, message.version
            )
        return True

    def _update_node_address(self, message: comm.NodeAddress):
        self._job_manager.update_node_service_addr(
            node_type=message.type,
            node_id=message.id,
            service_addr=message.addr,
        )
        return True

    def _deal_with_reported_node_event(self, message: comm.NodeEvent):
        node = Node(
            node_type=message.node.type,
            node_id=message.node.id,
            rank_index=message.node.rank,
        )
        event = NodeEvent(message.event_type, node)

        # let rdzv manager deal with rendezvous issue
        if event.is_node_check_event():
            net_rdzv_manager = self._rdzv_managers.get(
                RendezvousName.NETWORK_CHECK, None
            )
            if net_rdzv_manager:
                succeed = (
                    event.event_type == NodeEventType.NODE_CHECK_SUCCEEDED
                )
                net_rdzv_manager.report_network_check_result(
                    node.rank_index, succeed, message.event_elapsed_time
                )

        # let job manager deal with node issue(update status)
        self._job_manager.process_reported_node_event(event)
        return True

    def _handle_reported_atorch_event(self, message: comm.AtorchEvent):
        if message.name == TrainEventName.TRAIN_EVT_STEP:
            logger.debug(f"Add step event: {message}")
            _event_context.train_steps.add_step_event(message)
        elif message.name == TrainEventName.TRAIN_EVT_FLASH_CKPT:
            logger.debug(f"Add ckpt event: {message}")
            _event_context.ckpt_steps.add_ckpt_event(message)

        return True

    def _join_sync(self, node_type, node_id, message: comm.SyncJoin):
        success = False
        if self._sync_service:
            success = self._sync_service.join_sync(
                message.sync_name, node_type, node_id
            )
        return success

    def _sync_finished(self, message: comm.SyncFinish):
        success = False
        if self._sync_service:
            success = self._sync_service.sync_finished(message.sync_name)
        return success

    def _barrier(self, message: comm.SyncBarrier):
        if not self._sync_service:
            return False
        if message.notify:
            success = self._sync_service.notify_barrier(message.barrier_name)
        else:
            success = self._sync_service.barrier(message.barrier_name)
        return success

    def _report_rdzv_params(self, message: comm.RendezvousParams):
        # Enable auto-scaling workers if elasticity is enabled.
        for manager in self._rdzv_managers.values():
            manager.update_rdzv_params(
                min_nodes=message.min_nodes,
                max_nodes=message.max_nodes,
                waiting_timeout=message.waiting_timeout,
                node_unit=message.node_unit,
            )

        join_timeout = message.join_timeout
        if join_timeout == 0:  # Back compatibility
            join_timeout = JobConstant.RDZV_JOIN_TIMEOUT_DEFAULT
        self._job_manager.update_node_required_info(
            message.min_nodes, message.max_nodes, join_timeout
        )
        return True

    def _report_failure(self, node_type, node_id, message: comm.NodeFailure):
        self._job_manager.handle_training_failure(
            node_type,
            node_id,
            message.restart_count,
            message.error_data,
            message.level,
        )
        if message.level == TrainingExceptionLevel.RDZV_ERROR:
            custom_data = {
                CustomMetricKeys.TRAINING_ERROR_LEVEL: message.level,
                CustomMetricKeys.ERROR_CONTENT: message.error_data,
            }
            if self._job_metric_collector:
                self._job_metric_collector.collect_custom_data(custom_data)

            rdzv_error_data = json.loads(message.error_data)
            rdzv_name = rdzv_error_data["rdzv_name"]
            node_rank = rdzv_error_data["node_rank"]
            err_type = rdzv_error_data["err_type"]
            err_message = rdzv_error_data["err_message"]
            elapsed_time = rdzv_error_data["elapsed_time"]

            self._rdzv_managers[rdzv_name].process_error(
                node_id, node_rank, err_type, err_message, elapsed_time
            )

        return True

    def _kv_store_set(self, message: comm.KeyValuePair):
        self._kv_store.set(message.key, message.value)
        logger.debug(f"_kv_store_set: {message}")
        return True

    def _kv_store_multi_set(self, message: comm.KeyValuePairs):
        for k, v in message.kvs.items():
            self._kv_store.set(k, v)
        logger.debug(f"_kv_store_multi_set: {message}")
        return True

    def _report_paral_config(
        self, node_type, node_id, message: comm.ParallelConfig
    ):
        if self._job_manager:
            logger.debug(
                "Update parallel config for %s-%s: %s",
                node_type,
                node_id,
                message,
            )
            self._job_manager.update_node_paral_config(
                node_type, node_id, message
            )
        return True

    def _sync_checkpoint(
        self, node_type, node_id, message: comm.NodeCheckpointState
    ):
        if RendezvousName.TRAINING not in self._rdzv_managers:
            return False
        rdzv_manager = self._rdzv_managers[RendezvousName.TRAINING]
        return rdzv_manager.sync_ckpt_nodes(node_id, message.step)

    def _report_node_diagnosis_data(self, message: comm.DiagnosisReportData):
        if self._diagnosis_manager:
            data_cls: Optional[DiagnosisData] = getattr(
                self._diagnosis_data_module,
                message.data_cls,
            )
            if data_cls is None:
                logger.warning(
                    f"Invalid diagnosis report data type: {message.data_cls}"
                )
                return False
            data_obj = data_cls.from_json(message.data_content)
            self._diagnosis_manager.collect_diagnosis_data(data_obj)
        return True

    def _sync_training_ports(
        self, node_id, message: comm.SyncTrainingPort
    ) -> comm.SyncTrainingPort:
        logger.info(f"try to sync port {message.port} from {node_id}")
        sync_ports: SyncNodeTrainingPorts = (
            self._job_manager.sync_node_training_port(node_id, message.port)
        )
        return comm.SyncTrainingPort(
            port=sync_ports.training_port, newport=sync_ports.next_check_port
        )

    def _get_pre_check_result(
        self, node_type, node_id, message: comm.PreCheckRequest
    ) -> comm.PreCheckResponse:
        return comm.PreCheckResponse(
            status=get_job_context().get_pre_check_status()
        )

    def _report_event(self, message: comm.Event):
        if self._event_reporter:
            self._event_reporter.report(
                message.event_type,
                message.instance,
                message.action,
                message.msg,
                message.labels,
            )
        return True

    def _report_heartbeat(
        self, node_type, node_id, message: comm.HeartBeat
    ) -> comm.HeartbeatResponse:
        action = self._job_manager.collect_node_heart_beat(
            node_type, node_id, message.timestamp
        )
        if action and not isinstance(action, NoAction):
            logger.info(
                f"Master return action {action.__class__.__name__}: {action.to_json()}"
            )
        grpc_action = comm.DiagnosisAction(
            action.__class__.__name__,
            action.to_json(),
        )
        return comm.HeartbeatResponse(action=grpc_action)

    def _report_action(self, message: comm.DiagnosisAction):
        if not message:
            return False
        action_cls = message.action_cls
        action_content = message.action_content

        if action_cls == JobRestartAction.__name__:
            action = JobRestartAction.from_json(action_content)
        elif action_cls == JobAbortionAction.__name__:
            action = JobAbortionAction.from_json(action_content)
        else:
            # not supported for other type actions
            return False

        if action:
            _job_ctx.enqueue_diagnosis_action(action)
            return True

        return False


class RayMasterServicer(MasterServicer):
    """Master service with ray implementation."""

    def __init__(
        self,
        task_manager,
        job_manager,
        perf_monitor: PerfMonitor,
        rdzv_managers: Dict[str, RendezvousManager],
        diagnosis_manager: DiagnosisMaster,
        job_metric_collector=None,
        elastic_ps_service=None,
        sync_service=None,
    ):
        super().__init__(
            task_manager,
            job_manager,
            perf_monitor,
            rdzv_managers,
            diagnosis_manager,
            job_metric_collector,
            elastic_ps_service,
            sync_service,
        )

    def agent_report(self, request):
        return self.report(BaseRequest.from_json(request), None)

    def agent_get(self, request):
        return self.get(BaseRequest.from_json(request), None)

    def get_response(self, method):
        return BaseResponse()

    def get_task_type(self, task_type):
        return task_type


try:
    from dlrover.proto import elastic_training_pb2, elastic_training_pb2_grpc

    class GrpcMasterServicer(
        MasterServicer, elastic_training_pb2_grpc.MasterServicer
    ):
        """Master service with grpc implementation."""

        def __init__(
            self,
            task_manager,
            job_manager,
            perf_monitor: PerfMonitor,
            rdzv_managers: Dict[str, RendezvousManager],
            diagnosis_manager: DiagnosisMaster,
            job_metric_collector=None,
            elastic_ps_service=None,
            sync_service=None,
        ):
            super(GrpcMasterServicer, self).__init__(
                task_manager,
                job_manager,
                perf_monitor,
                rdzv_managers,
                diagnosis_manager,
                job_metric_collector,
                elastic_ps_service,
                sync_service,
            )

        def get(self, request, context: ServicerContext):
            request_meta = dict(context.invocation_metadata())

            if not self.validate_request(request_meta):
                context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                context.set_details(
                    CommunicationReqMeta.COMM_META_JOB_UID_INVALID_MSG
                )
                return self.get_response(
                    CommunicationReqType.COMM_REQ_TYPE_GET
                )

            return super().get(request, context)

        def report(self, request, context: ServicerContext):
            request_meta = dict(context.invocation_metadata())

            if not self.validate_request(request_meta):
                return self.get_response(
                    CommunicationReqType.COMM_REQ_TYPE_REPORT
                )

            return super().report(request, context)

        def get_response(self, method):
            if method == CommunicationReqType.COMM_REQ_TYPE_REPORT:
                return elastic_training_pb2.Response()
            else:
                return elastic_training_pb2.Message()

        def get_task_type(self, task_type):
            if task_type == TaskType.WAIT:
                return elastic_training_pb2.WAIT
            elif task_type == TaskType.TRAINING:
                return elastic_training_pb2.TRAINING
            elif task_type == TaskType.EVALUATION:
                return elastic_training_pb2.EVALUATION
            elif task_type == TaskType.PREDICTION:
                return elastic_training_pb2.PREDICTION
            elif task_type == TaskType.TRAIN_END_CALLBACK:
                return elastic_training_pb2.TRAIN_END_CALLBACK
            else:
                return elastic_training_pb2.NONE

except ImportError:
    logger.warning(
        "Protobuf is not installed. Can be ignored if "
        "using ray or using http server on k8s."
    )


try:
    import tornado
    from dlrover.python.common.http_server import TornadoHTTPServer

    class HttpMasterServicer(MasterServicer):
        """Master service with http implementation."""

        def __init__(
            self,
            task_manager,
            job_manager,
            perf_monitor: PerfMonitor,
            rdzv_managers: Dict[str, RendezvousManager],
            diagnosis_manager: DiagnosisMaster,
            job_metric_collector=None,
            elastic_ps_service=None,
            sync_service=None,
        ):
            super().__init__(
                task_manager,
                job_manager,
                perf_monitor,
                rdzv_managers,
                diagnosis_manager,
                job_metric_collector,
                elastic_ps_service,
                sync_service,
            )

        def get_response(self, method):
            return BaseResponse()

        def get_task_type(self, task_type):
            return task_type

    class HttpMasterHandler(tornado.web.RequestHandler):
        def initialize(self, master_servicer: HttpMasterServicer):
            self._handler = master_servicer

        def get(self):
            self.write("Not supported")

        def post(self):
            try:
                header = self.request.headers
                if not self._handler.validate_request(header):
                    self.set_status(406)
                    self.write(
                        CommunicationReqMeta.COMM_META_JOB_UID_INVALID_MSG
                    )
                    return

                path = self.request.path
                request = BaseRequest.from_json(json.loads(self.request.body))

                if path == "/get":
                    # return message
                    response = self._handler.get(request, BaseRequest())
                    if not response.data:
                        response.success = True
                    self.write(response.serialize())
                elif path == "/report":
                    # return boolean
                    self.write(
                        self._handler.report(
                            request, BaseRequest()
                        ).serialize()
                    )
                else:
                    self.set_status(404)
                    logger.error(f"No service found for {path}.")
                    self.write("")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self.set_status(500)
                self.write(f"{str(e)}")

except ImportError:
    logger.warning(
        "Tornado is not installed. Can be ignored if using ray or "
        "using grpc server on k8s."
    )


def create_master_service(
    port,
    task_manager,
    job_manager,
    perf_monitor,
    rdzv_managers,
    diagnosis_manager,
    job_metric_collector,
    elastic_ps_service,
    sync_service,
    max_threads=64,
):
    service_type = _dlrover_context.master_service_type
    logger.info(f"Creating master {service_type} service with port: {port}")

    server = None
    if service_type == CommunicationType.COMM_SERVICE_GRPC:
        import grpc as grpc_lib

        server = grpc_lib.server(
            futures.ThreadPoolExecutor(
                max_workers=max_threads,
                thread_name_prefix="grpc_master_service",
            ),
            options=[
                ("comm.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "comm.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        master_servicer = GrpcMasterServicer(
            task_manager=task_manager,
            job_manager=job_manager,
            perf_monitor=perf_monitor,
            rdzv_managers=rdzv_managers,
            diagnosis_manager=diagnosis_manager,
            job_metric_collector=job_metric_collector,
            elastic_ps_service=elastic_ps_service,
            sync_service=sync_service,
        )

        elastic_training_pb2_grpc.add_MasterServicer_to_server(
            master_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(port))
    elif service_type == CommunicationType.COMM_SERVICE_HTTP:
        master_servicer = HttpMasterServicer(
            task_manager=task_manager,
            job_manager=job_manager,
            perf_monitor=perf_monitor,
            rdzv_managers=rdzv_managers,
            diagnosis_manager=diagnosis_manager,
            job_metric_collector=job_metric_collector,
            elastic_ps_service=elastic_ps_service,
            sync_service=sync_service,
        )
        server = TornadoHTTPServer(
            "localhost",
            port,
            [
                (
                    r"/",
                    HttpMasterHandler,
                    dict(master_servicer=master_servicer),
                ),
                (
                    r"/get",
                    HttpMasterHandler,
                    dict(master_servicer=master_servicer),
                ),
                (
                    r"/report",
                    HttpMasterHandler,
                    dict(master_servicer=master_servicer),
                ),
            ],
        )

    return server
