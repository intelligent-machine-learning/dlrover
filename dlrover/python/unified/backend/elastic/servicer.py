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
from typing import List, Optional

from dlrover.python.common import comm
from dlrover.python.common.comm import BaseRequest, BaseResponse
from dlrover.python.common.constants import (
    CustomMetricKeys,
    NodeEventType,
    TrainingExceptionLevel,
)
from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.event.train_event import TrainEventName
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeEvent
from dlrover.python.diagnosis.common.diagnosis_action import NoAction
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.training_node import SyncNodeTrainingPorts
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.master.watcher.base_watcher import Node
from dlrover.python.util.queue.queue import RayEventQueue

from .manager import ElasticManager

ray_event_queue = RayEventQueue.singleton_instance()
_event_context = JobEventContext.singleton_instance()
job_ctx = get_job_context()


class MasterServicer(ABC):
    """Master service base class."""

    def __init__(
        self,
        job_manager: ElasticManager,
        job_metric_collector: Optional[JobMetricCollector] = None,
    ):
        self._core = job_manager
        self._job_metric_collector = job_metric_collector
        self._event_reporter = get_event_reporter()

        # preload module for class reflection
        self._diagnosis_data_module = importlib.import_module(
            "dlrover.python.diagnosis.common.diagnosis_data"
        )

    def get(self, request, _=None):
        node_type = request.node_type
        node_id = request.node_id
        req_message = comm.deserialize_message(request.data)

        response = BaseResponse()
        if not req_message:
            return response
        message = None
        if isinstance(req_message, comm.TaskRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.ShardCheckpointRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.ClusterVersionRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.RunningNodesRequest):
            message = self._get_running_nodes()
        elif isinstance(req_message, comm.JoinRendezvousRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.WaitingNodeNumRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.NetworkReadyRequest):
            message = self._check_fault_node()
        elif isinstance(req_message, comm.StragglerExistRequest):
            message = self._check_straggler()
        elif isinstance(req_message, comm.CommWorldRequest):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.KeyValuePair):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.KeyValuePairs):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(req_message, comm.PsNodesRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.TrainingStatusRequest):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(req_message, comm.ParallelConfigRequest):
            message = self._get_paral_config()
        elif isinstance(req_message, comm.CheckHardwareResetRequest):
            message = self._need_to_restart_training(node_type, node_id)
        elif isinstance(req_message, comm.SyncTrainingPort):
            message = self._sync_training_ports(node_id, req_message)
        elif isinstance(req_message, comm.ElasticRunConfigRequest):
            raise NotImplementedError("deprecated, useless in unified")
        elif isinstance(req_message, comm.PreCheckRequest):
            raise NotImplementedError("deprecated, useless in unified")
        elif isinstance(req_message, comm.HeartBeat):
            # raise NotImplementedError("deprecated, useless in unified")
            message = self._report_heartbeat(node_type, node_id, req_message)

        if message:
            response.data = message.serialize()
        return response

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

    def _check_fault_node(self):
        nodes, reason = self._core.node_check_manager.check_fault_node()
        res = comm.NetworkCheckResult(nodes=nodes, reason=reason)
        return res

    def _check_straggler(self):
        nodes, reason = self._core.node_check_manager.get_straggler()
        res = comm.NetworkCheckResult(nodes=nodes, reason=reason)
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

        response = BaseResponse()
        if not message:
            return response

        success = False
        if isinstance(message, comm.DatasetShardParams):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.ResourceStats):
            success = self._update_node_resource_usage(
                node_type, node_id, message
            )
        elif isinstance(message, comm.ModelInfo):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.GlobalStep):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.ShardCheckpoint):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.TaskResult):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.ClusterVersion):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.NodeAddress):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.NodeEvent):
            success = self._deal_with_reported_node_event(message)
        elif isinstance(message, comm.AtorchEvent):
            success = self._handle_reported_atorch_event(message)
        elif isinstance(message, comm.SyncJoin):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.SyncFinish):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.SyncBarrier):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.NodeFailure):
            success = self._report_failure(node_type, node_id, message)
        elif isinstance(message, comm.RendezvousParams):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(message, comm.PsReady):
            raise NotImplementedError("deprecated, TF backend only")
        elif isinstance(message, comm.KeyValuePair):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
        elif isinstance(message, comm.KeyValuePairs):
            raise NotImplementedError("deprecated, new RDZV based on Ray")
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
        sync_ports: SyncNodeTrainingPorts = self._core.sync_node_training_port(  # type:ignore #TODO
            node_id, message.port
        )
        return comm.SyncTrainingPort(
            port=sync_ports.training_port, newport=sync_ports.next_check_port
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
