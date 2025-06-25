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

import importlib
from abc import ABC
from typing import Dict, List, Optional

from dlrover.python.common import comm
from dlrover.python.common.comm import BaseRequest, BaseResponse
from dlrover.python.common.constants import (
    CustomMetricKeys,
    JobConstant,
    JobStage,
    KeyValueOps,
    NodeEventType,
    RendezvousName,
    TrainingExceptionLevel,
)
from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.event.train_event import TrainEventName
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeEvent
from dlrover.python.diagnosis.common.diagnosis_action import NoAction
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.hybrid.elastic.manager import ElasticManager
from dlrover.python.master.elastic_training.kv_store_service import (
    KVStoreService,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.training_node import SyncNodeTrainingPorts
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.watcher.base_watcher import Node
from dlrover.python.util.queue.queue import RayEventQueue

try:
    from dlrover.python.master.elastic_training.sync_service import SyncService
except ImportError:
    logger.info("Run the master locally.")
    pass


_dlrover_context = Context.singleton_instance()
ray_event_queue = RayEventQueue.singleton_instance()
_event_context = JobEventContext.singleton_instance()
job_ctx = get_job_context()


class MasterServicer(ABC):
    """Master service base class."""

    def __init__(
        self,
        job_manager: ElasticManager,
        job_metric_collector: Optional[JobMetricCollector] = None,
        sync_service: Optional[SyncService] = None,
    ):
        self._core = job_manager
        self._kv_store = KVStoreService()
        self._job_metric_collector = job_metric_collector
        self._sync_service = sync_service
        self._start_autoscale = False
        self._event_reporter = get_event_reporter()
        self._rdzv_managers = {
            RendezvousName.TRAINING: job_manager.rdzv_manager,
            RendezvousName.NETWORK_CHECK: job_manager.node_check_manager,
        }

        # preload module for class reflection
        self._diagnosis_data_module = importlib.import_module(
            "dlrover.python.diagnosis.common.diagnosis_data"
        )
        # clear kv store in case previous data is still there
        self._kv_store.clear()

    def get_response(self, method):
        """Should be implemented by subclasses."""
        return BaseResponse()

    def get_task_type(self, task_type):
        """Should be implemented by subclasses."""
        return task_type

    def get(self, request, _=None):
        node_type = request.node_type
        node_id = request.node_id
        req_message = comm.deserialize_message(request.data)

        response = self.get_response("get")
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
            configs = self._core.get_elastic_run_configs()
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
        raise NotImplementedError("deprecated, TF backend only")

    def _get_shard_checkpoint(self, request: comm.ShardCheckpointRequest):
        raise NotImplementedError("deprecated, TF backend only")

    def _get_cluster_version(self, request: comm.ClusterVersionRequest):
        return comm.ClusterVersion()

    def _query_ps_nodes(self):
        raise NotImplementedError("deprecated, TF backend only")

    def _get_running_nodes(self):
        res = comm.RunningNodes(nodes=[])
        nodes: List[Node] = self._core.get_running_nodes()
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
        raise NotImplementedError("deprecated, TF backend only")

    def _check_fault_node(self):
        nodes, reason = self._core.node_check_manager.check_fault_node()
        res = comm.NetworkCheckResult(nodes=nodes, reason=reason)
        return res

    def _check_straggler(self):
        nodes, reason = self._core.node_check_manager.get_straggler()
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
            self._core.rdzv_manager.clear_waiting_nodes()

        # Pause hang diagnosis during rendezvous
        if node_rank == 0:
            self._core.diagnosis.pause_observing()

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
        if job_ctx.get_job_stage() == JobStage.JOB_STOPPING:
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
            self._core.diagnosis.continue_observing()

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
        res = self._core.get_opt_strategy()
        if not res:
            res = comm.ParallelConfig()
        return res

    def _need_to_restart_training(self, node_type, node_id):
        restart = self._core.verify_restarting_worker_training(
            node_type, node_id
        )
        res = comm.ParallelConfig()
        res.restart = restart
        return res

    def report(self, request, _=None):
        node_type = request.node_type
        node_id = request.node_id
        message = comm.deserialize_message(request.data)

        response = self.get_response("report")
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

        response.success = success
        return response

    def _ready_for_ps_relaunch(self):
        raise NotImplementedError("deprecated, TF backend only")

    def _collect_dataset_shard_params(self, metrics: comm.DatasetShardParams):
        raise NotImplementedError("deprecated, TF backend only")

    def _update_node_resource_usage(
        self, node_type, node_id, metrics: comm.ResourceStats
    ):
        logger.debug(
            f"Update resource usage for {node_type}-{node_id},"
            f"cpu={metrics.cpu}, memory={metrics.memory},"
            f"gpu_stats={metrics.gpu_stats}"
        )
        if self._core:
            self._core.update_node_resource_usage(
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
        self._core.perf_monitor.collect_global_step(
            metrics.step, metrics.timestamp
        )
        if self._job_metric_collector and self._core:
            nodes = self._core.get_running_nodes()
            self._job_metric_collector.collect_runtime_stats(
                self._core.perf_monitor, nodes
            )
        self._check_start_auto_scale_worker()
        return True

    def _restore_shard_checkpoint(self, message: comm.ShardCheckpoint):
        raise NotImplementedError("deprecated, TF backend only")

    def _report_task_result(self, request: comm.TaskResult):
        raise NotImplementedError("deprecated, TF backend only")

    def _check_start_auto_scale_worker(self):
        sample_count = self._core.perf_monitor.get_sample_count()
        if (
            not self._start_autoscale
            and sample_count >= _dlrover_context.sample_count_to_adjust_worker
        ):
            logger.info(
                "Start autoscale with %s stats samples",
                sample_count,
            )
            self._core.start_auto_scaling()
            self._start_autoscale = True

    def _update_cluster_version(self, message: comm.ClusterVersion):
        raise NotImplementedError("deprecated, TF backend only")

    def _update_node_address(self, message: comm.NodeAddress):
        self._core.update_node_service_addr(
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
            succeed = event.event_type == NodeEventType.NODE_CHECK_SUCCEEDED
            self._core.node_check_manager.report_network_check_result(
                node.rank_index, succeed, message.event_elapsed_time
            )

        # let job manager deal with node issue(update status)
        self._core.process_reported_node_event(event)
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
        self._core.update_node_required_info(
            message.min_nodes, message.max_nodes, join_timeout
        )
        logger.info("debug rdzv return")
        return True

    def _report_failure(self, node_type, node_id, message: comm.NodeFailure):
        self._core.handle_training_failure(
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
        if self._core:
            logger.debug(
                "Update parallel config for %s-%s: %s",
                node_type,
                node_id,
                message,
            )
            self._core.update_node_paral_config(node_type, node_id, message)
        return True

    def _sync_checkpoint(
        self, node_type, node_id, message: comm.NodeCheckpointState
    ):
        return self._core.rdzv_manager.sync_ckpt_nodes(node_id, message.step)

    def _report_node_diagnosis_data(self, message: comm.DiagnosisReportData):
        if True:
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
            self._core.diagnosis.collect_diagnosis_data(data_obj)
        return True

    def _sync_training_ports(
        self, node_id, message: comm.SyncTrainingPort
    ) -> comm.SyncTrainingPort:
        logger.info(f"try to sync port {message.port} from {node_id}")
        sync_ports: SyncNodeTrainingPorts = self._core.sync_node_training_port(
            node_id, message.port
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
        action = self._core.collect_node_heart_beat(
            node_type, node_id, message.timestamp
        )
        if action and not isinstance(action, NoAction):
            logger.info(
                f"Master return action {action.__class__.__name__}: "
                f"{action.to_json()}"
            )
        grpc_action = comm.DiagnosisAction(
            action.__class__.__name__,
            action.to_json(),
        )
        return comm.HeartbeatResponse(action=grpc_action)


class RayMasterServicer(MasterServicer):
    """Master service with ray implementation."""

    def agent_report(self, request):
        return self.report(BaseRequest.from_json(request))

    def agent_get(self, request):
        return self.get(BaseRequest.from_json(request))
