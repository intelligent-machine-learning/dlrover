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
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple, Union

import requests

from dlrover.python.common import comm, env_utils
from dlrover.python.common.comm import BaseRequest, BaseResponse
from dlrover.python.common.constants import (
    CommunicationType,
    JobConstant,
    KeyValueOps,
    NetworkFailureReason,
    NodeEnv,
    NodeEventType,
    CommunicationReqMeta,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.util.common_util import find_free_port
from dlrover.python.util.function_util import retry


class MasterClient(Singleton, ABC):
    """MasterClient provides some APIs connect with the master
    service via gRPC/http/ray call.

    Args:
        master_addr: the master address
        node_id (int), the unique and ordered node ID assigned
        by dlrover command-line.
        node_type: the job type of node contains "worker", "ps"
            "evaluator" and "chief".
        timeout (int): the timeout second of requests.
    """

    _instance_lock = threading.Lock()

    def __init__(self, master_addr, node_id, node_type, timeout=5):
        logger.info(
            f"Build master client with master_addr: {master_addr}, "
            f"node_id: {node_id}, node_type: {node_type}."
        )
        self._timeout = timeout
        self._master_addr = master_addr
        self._node_id = node_id
        self._node_type = node_type
        self._node_ip = os.getenv("NODE_IP", "")
        self._worker_local_process_id = int(os.getenv("LOCAL_RANK", 0))
        self._ddp_server_port = find_free_port()
        self._diagnosis_action_module = importlib.import_module(
            "dlrover.python.diagnosis.common.diagnosis_action"
        )

    @retry()
    @abstractmethod
    def _report(self, message: comm.Message):
        """Abstraction of report function."""
        pass

    @retry()
    @abstractmethod
    def _get(self, message: comm.Message):
        """Abstraction of get function."""
        pass

    def kv_store_set(self, key, value):
        message = comm.KeyValuePair(key, value)
        message.op = KeyValueOps.SET
        response = self._report(message)
        logger.debug(f"kv_store_set: {message} {response}")
        return response.success

    def kv_store_get(self, key):
        request = comm.KeyValuePair(key)
        request.op = KeyValueOps.GET
        result: comm.KeyValuePair = self._get(request)
        logger.debug(f"kv_store_get: {request} {result}")
        return result.value

    def kv_store_add(self, key, value):
        request = comm.KeyValuePair(key, value)
        request.op = KeyValueOps.ADD
        result: comm.KeyValuePair = self._get(request)
        logger.debug(f"kv_store_add: {request} {result}")
        return result.value

    def kv_store_multi_get(self, keys):
        kvs = {key: b"" for key in keys}
        request = comm.KeyValuePairs(kvs)
        request.op = KeyValueOps.GET
        result: comm.KeyValuePairs = self._get(request)
        logger.debug(f"kv_store_multi_get: {request} {result}")
        return result.kvs

    def kv_store_multi_set(self, keys, values):
        try:
            kvs = {}
            for i in range(len(keys)):
                key = keys[i]
                value = values[i]
                kvs[key] = value
            message = comm.KeyValuePairs(kvs)
            message.op = KeyValueOps.SET
            response = self._report(message)
            logger.debug(f"kv_store_multi_set: {message} {response}")
            return response.success
        except IndexError:
            logger.warning(
                f"IndexError in kv_store_multi_set: {keys}, {values} are inconsistent"
            )
            raise
        except Exception:
            logger.warning(
                f"Unexpected error in kv_store_multi_set: {keys}, {values}"
            )
            raise

    def get_task(self, dataset_name):
        """Get a task from master.

        Args:
            dataset_name: string
            the training phase, c.f. /dlrover/proto/dlrover.proto

        Returns:
            the task unit assigned by master,
            c.f. /dlrover/proto/dlrover.proto
        """

        req = comm.TaskRequest(dataset_name)

        success = False
        res = None
        exception = None
        for _ in range(10):
            try:
                res = self._get(req)
                success = True
                break
            except Exception as e:
                exception = e
                time.sleep(15)
        if not success:
            logger.warning(exception)
        if not res:
            res = comm.Task()
        return success, res

    def report_task_result(self, dataset_name, task_id, err_msg):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.
        """
        message = comm.TaskResult(dataset_name, task_id, err_msg)
        return self._report(message)

    def report_dataset_shard_params(
        self,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        shuffle=False,
        num_minibatches_per_shard=0,
        dataset_name=None,
        task_type=0,
        storage_type="",
    ):
        message = comm.DatasetShardParams(
            batch_size=batch_size,
            num_epochs=num_epochs,
            dataset_size=dataset_size,
            shuffle=shuffle,
            num_minibatches_per_shard=num_minibatches_per_shard,
            dataset_name=dataset_name,
            task_type=task_type,
            storage_type=storage_type,
        )
        return self._report(message)

    def ready_for_ps_relaunch(self):
        message = comm.PsReady()
        return self._report(message)

    def get_shard_checkpoint(self, dataset_name):
        req = comm.ShardCheckpointRequest(dataset_name)
        res: comm.ShardCheckpoint = self._get(req)
        return res.content

    def report_shard_checkpoint(self, shard_checkpoint):
        request = comm.ShardCheckpoint(shard_checkpoint)
        return self._report(request)

    def report_used_resource(self, memory, cpu, gpu_stats):
        message = comm.ResourceStats(memory, cpu, gpu_stats)
        return self._report(message)

    def report_model_info(self, model_info):
        self._report(model_info)

    def report_global_step(
        self, global_step, timestamp, elapsed_time_per_step=0
    ):
        message = comm.GlobalStep(
            timestamp=timestamp,
            step=global_step,
            elapsed_time_per_step=elapsed_time_per_step,
        )
        return self._report(message)

    def report_heart_beat(self, timestamp) -> DiagnosisAction:
        message = comm.HeartBeat(timestamp=timestamp)
        response: comm.HeartbeatResponse = self._get(message)
        action = NoAction()

        if not response:
            logger.warning("No response from heartbeat reporting.")
            return action

        action_cls: Optional[DiagnosisData] = getattr(
            self._diagnosis_action_module,
            response.action.action_cls,
        )
        if action_cls is None:
            logger.warning(
                f"Invalid diagnosis action action type: {response.action.action_cls}"
            )
        else:
            action = action_cls.from_json(response.action.action_content)
        return action

    def update_node_addr(self, task_type, task_id, node_addr):
        message = comm.NodeAddress(type=task_type, id=task_id, addr=node_addr)
        res = self._report(message)
        return res

    def report_node_event(
        self,
        event_type,
        event_msg="",
        event_time=int(time.time()),
        event_elapsed_time=0,
        node_rank=-1,
    ):
        message = comm.NodeEvent(
            event_type=event_type,
            event_message=event_msg,
            event_time=event_time,
            event_elapsed_time=event_elapsed_time,
            node=comm.NodeMeta(
                type=self._node_type, id=self._node_id, addr=self._node_ip
            ),
        )

        if node_rank != -1:
            message.node.rank = node_rank

        return self._report(message)

    def report_atorch_event(
        self,
        event_ts,
        event_target,
        event_name,
        event_type,
        event_step,
    ):
        message = comm.AtorchEvent(
            timestamp=event_ts,
            step=event_step,
            target=event_target,
            name=event_name,
            type=event_type,
        )

        return self._report(message)

    def report_network_check_status(self, node_rank, status, elapsed_time):
        return self.report_node_event(
            event_type=status,
            event_elapsed_time=elapsed_time,
            node_rank=node_rank,
        )

    def report_pre_check_status(self, status, config_json="{}"):
        return self.report_node_event(
            event_type=status,
            event_msg=config_json,
            event_time=int(time.time()),
            event_elapsed_time=0,
            node_rank=env_utils.get_node_rank(),
        )

    def report_failed_exited(self):
        return self.report_node_event(NodeEventType.FAILED_EXITED)

    def report_succeeded_exited(self):
        return self.report_node_event(NodeEventType.SUCCEEDED_EXITED)

    def get_cluster_version(self, version_type, task_type, task_id):
        request = comm.ClusterVersionRequest(
            task_type=task_type,
            task_id=task_id,
            version_type=version_type,
        )
        result: comm.ClusterVersion = self._get(request)
        return result.version

    def update_cluster_version(
        self, version_type, version, task_type, task_id
    ):
        message = comm.ClusterVersion(
            task_type=task_type,
            task_id=task_id,
            version_type=version_type,
            version=version,
        )
        self._report(message)

    def query_ps_nodes(self):
        request = comm.PsNodesRequest()
        result: comm.PsNodes = self._get(request)
        return result.nodes, result.ps_failure

    def query_training_status(self):
        request = comm.TrainingStatusRequest()
        response: comm.TrainingStatus = self._get(request)
        return response.status

    def join_sync(self, sync_name):
        message = comm.SyncJoin(sync_name)
        logger.info(
            " {}:{} join sync {}".format(
                self._node_id, self._node_type, sync_name
            )
        )
        response = self._report(message)
        return response.success

    def sync_finished(self, sync_name):
        message = comm.SyncFinish(sync_name)
        response = self._report(message)
        return response.success

    def barrier(self, barrier_name, notify=False):
        message = comm.SyncBarrier(barrier_name, notify)
        response = self._report(message)
        return response.success

    def get_running_nodes(self):
        request = comm.RunningNodesRequest()
        result: comm.RunningNodes = self._get(request)
        return result.nodes

    def num_nodes_waiting(self, rdzv_name):
        request = comm.WaitingNodeNumRequest(rdzv_name=rdzv_name)
        try:
            result: comm.RendezvousState = self._get(request)
            return result.waiting_num
        except Exception:
            logger.warning("Fail to query the number of waiting nodes.")
            return 0

    def join_rendezvous(self, node_rank, local_world_size, rdzv_name=""):
        request = comm.JoinRendezvousRequest(
            node_id=self._node_id,
            node_rank=node_rank,
            local_world_size=local_world_size,
            rdzv_name=rdzv_name,
            node_ip=self._node_ip,
        )
        result: comm.RendezvousState = self._get(request)
        return result.round

    def get_comm_world(self, rdzv_name, node_rank):
        request = comm.CommWorldRequest(node_id=node_rank, rdzv_name=rdzv_name)
        result: comm.RendezvousState = self._get(request)
        return result.round, result.group, result.world

    def check_fault_node(self, timeout=300):
        request = comm.NetworkReadyRequest()
        start = time.time()
        while True:
            result: comm.NetworkCheckResult = self._get(request)
            if (
                result.reason == NetworkFailureReason.WAITING_NODE
                or result.reason == NetworkFailureReason.NO_INIT
            ) and time.time() - start < timeout:
                time.sleep(JobConstant.MASTER_CLIENT_CHECK_FAULT_SLEEP_TIMEOUT)
                continue
            break
        return result.nodes, result.reason

    def check_straggler(self, timeout=300):
        request = comm.StragglerExistRequest()
        start = time.time()
        while True:
            result: comm.NetworkCheckResult = self._get(request)
            if (
                result.reason == NetworkFailureReason.WAITING_NODE
                and time.time() - start < timeout
            ):
                time.sleep(
                    JobConstant.MASTER_CLIENT_CHECK_STRAGGLER_SLEEP_TIMEOUT
                )
                continue
            break
        return result.nodes, result.reason

    def report_rdzv_params(
        self, min_nodes, max_nodes, waiting_timeout, node_unit, joint_timeout
    ):
        message = comm.RendezvousParams(
            min_nodes,
            max_nodes,
            waiting_timeout,
            node_unit,
            joint_timeout,
        )
        response = self._report(message)
        return response.success

    def report_failures(self, error_data, restart_count=-1, level=""):
        message = comm.NodeFailure(error_data, restart_count, level)
        self._report(message)

    def report_paral_config(self, config: comm.ParallelConfig):
        self._report(config)

    def report_diagnosis_agent_metrics(self, data: DiagnosisData):
        message = comm.DiagnosisReportData(
            data.__class__.__name__,
            data.to_json(),
            data.node_rank,
        )
        self._report(message)

    def get_paral_config(self) -> comm.ParallelConfig:
        request = comm.ParallelConfigRequest()
        result = self._get(request)
        return result

    def need_to_restart_training(self):
        request = comm.CheckHardwareResetRequest()
        try:
            result: comm.ParallelConfig = self._get(request)
            return result.restart
        except Exception:
            logger.warning("Fail to verify restarting training processes.")
            return False

    def sync_checkpoint(self, step):
        request = comm.NodeCheckpointState()
        request.step = step
        response = self._report(request)
        return response.success

    def sync_training_ports(self, port) -> comm.SyncTrainingPort:
        request = comm.SyncTrainingPort(port=port)
        response: comm.SyncTrainingPort = self._get(request)
        return response

    def get_elastic_run_config(self) -> Dict[str, str]:
        request = comm.ElasticRunConfigRequest()
        response: comm.ElasticRunConfig = self._get(request)
        return response.configs

    def get_pre_check_result(self) -> str:
        request = comm.PreCheckRequest()
        response: comm.PreCheckResponse = self._get(request)
        return response.status

    def report_event(
        self,
        event_type: str = "",
        instance: str = "",
        action: str = "",
        msg: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        if labels is None:
            labels = {}
        message = comm.Event(
            event_type=event_type,
            instance=instance,
            action=action,
            msg=msg,
            labels=labels,
        )
        self._report(message)

    def set_rdzv_blocked(self, blocked, reason=""):
        message = comm.RdzvBlocked(blocked=blocked, reason=reason)
        self._report(message)

    def report_action(self, action: DiagnosisAction):
        message = comm.DiagnosisAction(
            action_cls=action.__class__.__name__,
            action_content=action.to_json(),
        )
        self._report(message)

    @classmethod
    def singleton_instance(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = build_master_client(*args, **kwargs)
        return cls._instance


try:
    from dlrover.proto import elastic_training_pb2, elastic_training_pb2_grpc

    class GrpcMasterClient(MasterClient):
        """
        Examples::
        channel = elasticai_api.util.grpc_utils.build_channel(
            "localhost:50001"
        )
        mc = MasterClient(channel, work_id=0)
        # get task unit from master service
        mc.get_task(...)
        """

        def __init__(self, master_addr, node_id, node_type, timeout=5):
            super(GrpcMasterClient, self).__init__(
                master_addr, node_id, node_type, timeout
            )
            self._open_grpc_channel()

        def __del__(self):
            self._close_grpc_channel()

        def _close_grpc_channel(self):
            if self._channel:
                self._channel.close()

        def _open_grpc_channel(self):
            self._channel = comm.build_grpc_channel(self._master_addr)
            self._stub = elastic_training_pb2_grpc.MasterStub(self._channel)

        def _gen_request_meta(self) -> Sequence[Tuple[str, Union[str, bytes]]]:
            return [
                (
                    CommunicationReqMeta.COMM_META_JOB_UID,
                    env_utils.get_job_uid(),
                )
            ]

        @retry()
        def _report(self, message: comm.Message):
            request = elastic_training_pb2.Message()
            request.node_id = self._node_id
            request.node_type = self._node_type
            request.data = message.serialize()
            return self._stub.report(
                request,
                timeout=self._timeout,
                metadata=self._gen_request_meta(),
            )

        @retry()
        def _get(self, message: comm.Message):
            request = elastic_training_pb2.Message()
            request.node_id = self._node_id
            request.node_type = self._node_type
            request.data = message.serialize()
            response = self._stub.get(
                request,
                timeout=self._timeout,
                metadata=self._gen_request_meta(),
            )
            res_message = comm.deserialize_message(response.data)
            return res_message

except ImportError:
    logger.warning("Protobuf is not installed.")


class HttpMasterClient(MasterClient):
    def __init__(self, master_addr, node_id, node_type, timeout=5):
        super(HttpMasterClient, self).__init__(
            master_addr, node_id, node_type, timeout
        )

    def _get_http_request_url(self, path: str) -> str:
        return "http://" + self._master_addr + path

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            CommunicationReqMeta.COMM_META_JOB_UID: env_utils.get_job_uid(),
        }

    @retry()
    def _report(self, message: comm.Message):
        with requests.post(
            self._get_http_request_url("/report"),
            json=self._gen_request(message).to_json(),
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                error_msg = f"Failed to report master with http request: {type(message)}."
                raise RuntimeError(error_msg)
            response_data: BaseResponse = comm.deserialize_message(
                response.content
            )
            return response_data

    @retry()
    def _get(self, message: comm.Message):
        with requests.post(
            self._get_http_request_url("/get"),
            json=self._gen_request(message).to_json(),
            headers=self._get_headers(),
        ) as response:
            if response.status_code != 200:
                error_msg = f"Failed to get from master with http request: {type(message)}."
                raise RuntimeError(error_msg)
            response_data: BaseResponse = comm.deserialize_message(
                response.content
            )
            return comm.deserialize_message(response_data.data)

    def _gen_request(self, message: comm.Message):
        request = BaseRequest()
        request.node_id = self._node_id
        request.node_type = self._node_type
        request.data = message.serialize()

        return request


try:
    import ray
    from ray.util.state import get_actor

    class RayMasterClient(MasterClient):
        def __init__(self, master_addr, node_id, node_type, timeout=5):
            super(RayMasterClient, self).__init__(
                master_addr, node_id, node_type, timeout
            )
            self._master_actor_handle = None

        def _get_master_actor_handle(self):
            if not self._master_actor_handle:
                self._master_actor_handle = ray.get_actor(
                    get_actor(self._master_addr).name
                )
            return self._master_actor_handle

        @retry()
        def _report(self, message: comm.Message):
            response = ray.get(
                self._get_master_actor_handle().agent_report.remote(
                    self._gen_request(message).to_json()
                ),
                timeout=self._timeout,
            )

            return response

        @retry()
        def _get(self, message: comm.Message):
            response = ray.get(
                self._get_master_actor_handle().agent_get.remote(
                    self._gen_request(message).to_json()
                ),
                timeout=self._timeout,
            )

            return comm.deserialize_message(response.data)

        def _gen_request(self, message: comm.Message):
            request = BaseRequest()
            request.node_id = self._node_id
            request.node_type = self._node_type
            request.data = message.serialize()

            return request

        def get_elastic_run_config(self) -> Dict[str, str]:
            # no need to get config from master
            return {}

except (ImportError, TypeError):
    logger.warning("Ray is not installed or deps not satisfied.")


def build_master_client(
    master_addr=None, timeout=JobConstant.MASTER_CLIENT_DEFAULT_TIMEOUT
):
    """
    Build a master client.

    Args:
        master_addr (Union[str, ActorHandle]): the address of the job master,
            the format is "{IP}:{PORT}" if str type
        timeout (int): the timeout second of grpc requests.
    """
    if master_addr is None:
        master_addr = os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "")
    node_id = env_utils.get_node_id()
    node_type = env_utils.get_node_type()

    try:
        _timeout = int(os.getenv(NodeEnv.MASTER_CLIENT_TIMEOUT, ""))
        logger.info(f"set master_client timeout to env {_timeout}")
    except Exception:
        _timeout = timeout
        logger.info(f"set master_client timeout to {_timeout}")

    master_client = None
    master_service_type = os.getenv(
        NodeEnv.DLROVER_MASTER_SERVICE_TYPE,
        CommunicationType.COMM_SERVICE_GRPC,
    )
    logger.info(f"Use [{master_service_type}] type for master client.")

    if master_addr:
        try:
            if master_service_type == CommunicationType.COMM_SERVICE_GRPC:
                master_client = GrpcMasterClient(
                    master_addr, node_id, node_type, timeout
                )
            elif master_service_type == CommunicationType.COMM_SERVICE_HTTP:
                master_client = HttpMasterClient(
                    master_addr, node_id, node_type, timeout
                )
            else:
                master_client = RayMasterClient(
                    master_addr, node_id, node_type, timeout
                )
        except Exception:
            logger.warning("The master is not available.")

    return master_client
