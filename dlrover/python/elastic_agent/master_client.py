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

import os
import random
import socket
import time
from contextlib import closing

from dlrover.proto import elastic_training_pb2, elastic_training_pb2_grpc
from dlrover.python.common import grpc
from dlrover.python.common.constants import NetworkFailureReason, NodeEnv
from dlrover.python.common.log import default_logger as logger


def retry_grpc_request(func):
    def wrapper(self, *args, **kwargs):
        retry = kwargs.get("retry", 10)
        execption = None
        for i in range(retry):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Retry %s to %s.%s with failure",
                    i,
                    self.__class__.__name__,
                    func.__name__,
                )
                execption = e
                time.sleep(5)
        if execption:
            logger.error(execption)
            raise execption

    return wrapper


class MasterClient(object):
    """MasterClient provides some APIs connect with the master
    service via gRPC call.

    Usage:
        channel = elasticai_api.util.grpc_utils.build_channel(
            "localhost:50001"
        )
        mc = MasterClient(channel, work_id=0)
        # get task unit from master service
        mc.get_task(...)
    """

    def __init__(self, master_addr, node_id, node_type):
        """Initialize a master client.
        Args:
            master_addr: the master address

            worker_id: int
            the unique and ordered worker ID assigned
            by dlrover command-line.
        """
        self._master_addr = master_addr
        self._channel = grpc.build_channel(master_addr)
        logger.info("dlrover master addr is %s" % self._master_addr)
        self._stub = elastic_training_pb2_grpc.MasterStub(self._channel)
        self._node_id = node_id
        self._node_type = node_type
        self._host = os.getenv("MY_POD_IP", "localhost")
        self._worker_local_process_id = int(os.getenv("LOCAL_RANK", 0))
        self._ddp_server_port = self.find_free_port()
        self._host_name = os.getenv("POD_NAME", "")

    def __del__(self):
        self._channel.close()

    def close_channel(self):
        self._channel.close()

    def open_channel(self):
        self._channel = grpc.build_channel(self._master_addr)
        self._stub = elastic_training_pb2_grpc.MasterStub(self._channel)

    def find_free_port(self):
        with closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", 0))
            _, port = sock.getsockname()
            return port

    @retry_grpc_request
    def _report(self, message: grpc.Message):
        request = elastic_training_pb2.Message()
        request.node_id = self._node_id
        request.node_type = self._node_type
        request.data = message.serialize()
        return self._stub.report(request, timeout=5)

    @retry_grpc_request
    def _get(self, message: grpc.Message):
        request = elastic_training_pb2.Message()
        request.node_id = self._node_id
        request.node_type = self._node_type
        request.data = message.serialize()
        response = self._stub.get(request, timeout=5)
        res_message = grpc.deserialize_message(response.data)
        return res_message

    def kv_store_set(self, key, value):
        message = grpc.KeyValuePair(key, value)
        response = self._report(message)
        return response.success

    def kv_store_get(self, key):
        request = grpc.KeyValuePair(key)
        result: grpc.KeyValuePair = self._get(request)
        return result.value

    def get_task(self, dataset_name):
        """Get a task from master.

        Args:
            dataset_name: string
            the training phase, c.f. /dlrover/proto/dlrover.proto

        Returns:
            the task unit assigned by master,
            c.f. /dlrover/proto/dlrover.proto
        """

        req = grpc.TaskRequest(dataset_name)

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
            res = grpc.Task()
        return success, res

    def report_task_result(self, dataset_name, task_id, err_msg):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.
        """
        message = grpc.TaskResult(dataset_name, task_id, err_msg)
        return self._report(message)

    def report_dataset_shard_params(
        self,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        shuffle=False,
        num_minibatches_per_shard=0,
        dataset_name=None,
        task_type=elastic_training_pb2.NONE,
        storage_type="",
    ):
        message = grpc.DatasetShardParams(
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
        message = grpc.PsReady()
        return self._report(message)

    def get_shard_checkpoint(self, dataset_name):
        req = grpc.ShardCheckpointRequest(dataset_name)
        res: grpc.ShardCheckpoint = self._get(req)
        return res.content

    def report_shard_checkpoint(self, shard_checkpoint):
        request = grpc.ShardCheckpoint(shard_checkpoint)
        return self._report(request)

    def report_used_resource(self, memory, cpu, gpu_stats):
        message = grpc.ResourceStats(memory, cpu, gpu_stats)
        return self._report(message)

    def report_model_info(self, model_info):
        self._report(model_info)

    def report_global_step(
        self, global_step, timestamp, elapsed_time_per_step=0
    ):
        message = grpc.GlobalStep(
            timestamp=timestamp,
            step=global_step,
            elapsed_time_per_step=elapsed_time_per_step,
        )
        return self._report(message)

    def get_cluster_version(self, version_type, task_type, task_id):
        request = grpc.ClusterVersionRequest(
            task_type=task_type,
            task_id=task_id,
            version_type=version_type,
        )
        result: grpc.ClusterVersion = self._get(request)
        return result.version

    def update_node_addr(self, task_type, task_id, node_addr):
        message = grpc.NodeAddress(type=task_type, id=task_id, addr=node_addr)
        res = self._report(message)
        return res

    def update_node_event(self, task_type, task_id, event):
        message = grpc.NodeEvent(
            event_type="1",
            message="train_success",
            node=grpc.NodeMeta(type=task_type, id=task_id),
        )
        return self._report(message)

    def update_cluster_version(
        self, version_type, version, task_type, task_id
    ):
        message = grpc.ClusterVersion(
            task_type=task_type,
            task_id=task_id,
            version_type=version_type,
            version=version,
        )
        self._report(message)

    def query_ps_nodes(self):
        request = grpc.PsNodesRequest()
        result: grpc.PsNodes = self._get(request)
        return result.nodes, result.ps_failure

    def query_training_status(self):
        request = grpc.TrainingStatusRequest()
        response: grpc.TrainingStatus = self._get(request)
        return response.status

    def join_sync(self, sync_name):
        message = grpc.SyncJoin(sync_name)
        logger.info(
            " {}:{} join sync {}".format(
                self._node_id, self._node_type, sync_name
            )
        )
        response = self._report(message)
        return response.success

    def sync_finished(self, sync_name):
        message = grpc.SyncFinish(sync_name)
        response = self._report(message)
        return response.success

    def barrier(self, barrier_name, notify=False):
        message = grpc.SyncBarrier(barrier_name, notify)
        response = self._report(message)
        return response.success

    def get_running_nodes(self):
        request = grpc.RunningNodesRequest()
        result: grpc.RunningNodes = self._get(request)
        return result.nodes

    def num_nodes_waiting(self, rdzv_name):
        request = grpc.WaitingNodeNumRequest(rdzv_name=rdzv_name)
        result: grpc.RendezvousState = self._get(request)
        return result.waiting_num

    def join_rendezvous(self, rank_id, local_world_size, rdzv_name=""):
        request = grpc.JoinRendezvousRequest(
            node_id=rank_id,
            local_world_size=local_world_size,
            rdzv_name=rdzv_name,
        )
        result: grpc.RendezvousState = self._get(request)
        return result.round

    def get_comm_world(self, rdzv_name, rank_id):
        request = grpc.CommWorldRequest(node_id=rank_id, rdzv_name=rdzv_name)
        result: grpc.RendezvousState = self._get(request)
        return result.group, result.world

    def check_fault_node(self, timeout=300):
        request = grpc.NetworkReadyRequest()
        start = time.time()
        while True:
            result: grpc.NetworkCheckResult = self._get(request)
            if (
                result.reason == NetworkFailureReason.WAITING_NODE
                and time.time() - start < timeout
            ):
                time.sleep(5)
                continue
            break
        return result.nodes

    def check_straggler(self, timeout=300):
        request = grpc.StragglerExistRequest()
        start = time.time()
        while True:
            result: grpc.NetworkCheckResult = self._get(request)
            if (
                result.reason == NetworkFailureReason.WAITING_NODE
                and time.time() - start < timeout
            ):
                time.sleep(5)
                continue
            break
        return result.nodes

    def report_rdzv_params(
        self, min_nodes, max_nodes, waiting_timeout, node_unit
    ):
        message = grpc.RendezvousParams(
            min_nodes,
            max_nodes,
            waiting_timeout,
            node_unit,
        )
        response = self._report(message)
        return response.success

    def report_network_status(self, rank_id, status, elasped_time):
        message = grpc.NetworkStatus(
            rank=rank_id, status=status, elasped_time=elasped_time
        )
        self._report(message)

    def report_failures(self, error_data, restart_count=-1, level=""):
        message = grpc.NodeFailure(error_data, restart_count, level)
        self._report(message)

    def report_paral_config(self, config: grpc.ParallelConfig):
        self._report(config)

    def get_paral_config(self) -> grpc.ParallelConfig:
        request = grpc.ParallelConfigRequest()
        result = self._get(request)
        return result


class LocalDataset(object):
    def __init__(
        self,
        batch_size,
        num_epochs,
        dataset_size,
        shuffle,
        num_minibatches_per_shard,
        task_type=elastic_training_pb2.NONE,
    ):
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._dataset_size = dataset_size
        self._shuffle = shuffle
        self._records_per_shard = batch_size * num_minibatches_per_shard
        self._todo = []
        self._epoch = 0
        self._task_type = task_type

    def create_tasks(self):
        start = 0
        if self._records_per_shard <= 0:
            raise ValueError(
                "records_per_shard {} can not be less than 1".format(
                    self._records_per_shard
                )
            )
        while start < self._dataset_size:
            end = min(start + self._records_per_shard, self._dataset_size)
            self._todo.append((start, end))
            start = end
        if self._shuffle:
            random.shuffle(self._todo)

    def get_task(self, task_type=None):
        start = -1
        end = -1
        if not self._todo and self._epoch < self._num_epochs:
            self.create_tasks()
            self._epoch += 1
        if self._todo:
            start, end = self._todo.pop(0)
        return start, end

    def get_current_epoch(self):
        return self._epoch

    def reset(self):
        self._epoch = 0
        self._todo = []


def build_master_client(master_addr=None):
    if master_addr is None:
        master_addr = os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "")
    worker_id = int(os.getenv(NodeEnv.WORKER_ID, 0))
    worker_type = os.getenv(NodeEnv.WORKER_TYPE, "worker")

    master_client = None
    logger.info(f"Build master client with addr {master_addr}.")
    if master_addr:
        master_client = MasterClient(master_addr, worker_id, worker_type)
    return master_client


class GlobalMasterClient(object):
    MASTER_CLIENT = build_master_client()
