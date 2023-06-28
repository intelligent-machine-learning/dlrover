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
from typing import Dict

from google.protobuf import empty_pb2

from dlrover.proto import elastic_training_pb2, elastic_training_pb2_grpc
from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.grpc import build_channel
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
        self._channel = build_channel(master_addr)
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
        self._channel = build_channel(self._master_addr)
        self._stub = elastic_training_pb2_grpc.MasterStub(self._channel)

    def find_free_port(self):
        with closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", 0))
            _, port = sock.getsockname()
            return port

    def get_task(self, dataset_name):
        """Get a task from master.

        Args:
            dataset_name: string
            the training phase, c.f. /dlrover/proto/dlrover.proto

        Returns:
            the task unit assigned by master,
            c.f. /dlrover/proto/dlrover.proto
        """

        req = elastic_training_pb2.GetTaskRequest()
        req.worker_type = self._node_type
        req.worker_id = self._node_id
        req.dataset_name = dataset_name

        success = False
        res = None
        exception = None
        for _ in range(10):
            try:
                res = self._stub.get_task(req)
                success = True
                break
            except Exception as e:
                exception = e
                time.sleep(15)
        if not success:
            logger.warning(exception)
        if not res:
            res = elastic_training_pb2.Task()
        return success, res

    @retry_grpc_request
    def report_task_result(self, dataset_name, task_id, err_msg):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.
        """
        request = elastic_training_pb2.ReportTaskResultRequest()
        request.dataset_name = dataset_name
        request.task_id = task_id
        request.err_message = err_msg
        return self._stub.report_task_result(request)

    @retry_grpc_request
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
        request = elastic_training_pb2.ReportDatasetShardParamsRequest()
        request.batch_size = batch_size
        request.shuffle = shuffle
        request.task_type = task_type
        request.dataset_name = dataset_name
        if num_epochs is not None:
            request.num_epochs = num_epochs
        if dataset_size is not None:
            request.dataset_size = dataset_size
        request.num_minibatches_per_shard = num_minibatches_per_shard
        request.storage_type = storage_type
        return self._stub.report_dataset_shard_params(request)

    @retry_grpc_request
    def ready_for_ps_relaunch(self):
        request = empty_pb2.Empty()
        self._stub.ready_for_ps_relaunch(request)

    @retry_grpc_request
    def get_shard_checkpoint(self, dataset_name):
        request = elastic_training_pb2.DatasetMeta()
        request.dataset_name = dataset_name if dataset_name else ""
        return self._stub.get_shard_checkpoint(request)

    @retry_grpc_request
    def report_shard_checkpoint(self, shard_checkpoint):
        request = elastic_training_pb2.ShardCheckpoint()
        request.content = shard_checkpoint
        return self._stub.report_shard_checkpoint(request)

    @retry_grpc_request
    def report_used_resource(self, memory, cpu):
        request = elastic_training_pb2.ReportUsedResourceRequest()
        request.memory = memory
        request.cpu = cpu
        request.node_id = self._node_id
        request.node_type = self._node_type
        return self._stub.report_used_resource(request)

    @retry_grpc_request
    def get_dataset_epoch(self, dataset_name):
        request = elastic_training_pb2.DatasetMeta()
        request.dataset_name = dataset_name if dataset_name else ""
        return self._stub.get_dataset_epoch(request)

    @retry_grpc_request
    def report_model_metric(self, tensor_stats, op_stats):
        metric_msg = elastic_training_pb2.ModelMetric()
        tensor_msg = metric_msg.tensor_stats
        tensor_msg.variable_count = tensor_stats.variable_count
        tensor_msg.total_variable_size = tensor_stats.total_variable_size
        tensor_msg.max_variable_size = tensor_stats.max_variable_size
        tensor_msg.tensor_alloc_bytes.update(tensor_stats.tensor_alloc_bytes)
        tensor_msg.kv_embedding_dims.extend(tensor_stats.kv_embedding_dims)

        op_msg = metric_msg.op_stats
        op_msg.op_count = op_stats.op_count
        op_msg.update_op_count = op_stats.update_op_count
        op_msg.read_op_count = op_stats.read_op_count
        op_msg.input_fetch_dur = op_stats.input_fetch_dur
        op_msg.flops = op_stats.flops
        op_msg.recv_op_count = op_stats.recv_op_count
        return self._stub.report_model_metric(metric_msg)

    @retry_grpc_request
    def report_global_step(self, global_step, timestamp):
        record = elastic_training_pb2.GlobalStepRecord()
        record.global_step = global_step
        record.timestamp = timestamp
        return self._stub.report_global_step(record, timeout=10)

    @retry_grpc_request
    def get_cluster_version(self, version_type, task_type, task_id):
        request = elastic_training_pb2.GetClusterVersionRequest()
        request.task_id = task_id
        request.version_type = version_type
        request.task_type = task_type
        return self._stub.get_cluster_version(request)

    def update_node_addr(self, task_type, task_id, node_addr):
        request = elastic_training_pb2.NodeMeta()
        request.id = task_id
        request.type = task_type
        request.addr = node_addr
        request.rank = -1
        res = self._stub.update_node_status(request)
        return res

    @retry_grpc_request
    def update_node_event(self, task_type, task_id, event):
        request = elastic_training_pb2.NodeEvent()
        request.node.id = task_id
        request.node.type = task_type
        request.message = "train_success"
        request.event_type = "1"
        return self._stub.update_node_event(request)

    @retry_grpc_request
    def update_cluster_version(
        self, version_type, version, task_type, task_id
    ):
        request = elastic_training_pb2.UpdateClusterVersionRequest()
        request.task_id = task_id
        request.version_type = version_type
        request.version = version
        request.task_type = task_type
        self._stub.update_cluster_version(request)

    def query_ps_nodes(self):
        request = empty_pb2.Empty()
        response = self._stub.query_ps_nodes(request)
        return response.ps_nodes, response.ps_failure

    @retry_grpc_request
    def query_training_status(self):
        request = empty_pb2.Empty()
        response = self._stub.query_training_status(request)
        return response.status

    @retry_grpc_request
    def get_dataset_shard_num(self, dataset_name):
        request = elastic_training_pb2.DatasetMeta()
        request.dataset_name = dataset_name
        response = self._stub.get_dataset_shard_num(request)
        return response.shard_num

    @retry_grpc_request
    def join_sync(self, sync_name):
        request = elastic_training_pb2.SyncRequest()
        request.sync_name = sync_name
        request.worker_id = self._node_id
        request.worker_type = self._node_type
        logger.info(
            " {}:{} join sync {}".format(
                self._node_id, self._node_type, sync_name
            )
        )
        return self._stub.join_sync(request)

    @retry_grpc_request
    def sync_finished(self, sync_name):
        request = elastic_training_pb2.SyncRequest()
        request.sync_name = sync_name
        return self._stub.sync_finished(request)

    @retry_grpc_request
    def barrier(self, barrier_name, notify=False):
        request = elastic_training_pb2.BarrierRequest()
        request.barrier_name = barrier_name
        request.notify = notify
        return self._stub.barrier(request)

    def report_prestop(self):
        req = elastic_training_pb2.ReportPreStopRequest()
        req.worker_host = self._host
        logger.info("Worker {} report prestop hook".format(self._host))
        return self._stub.report_prestop(req)

    def get_running_nodes(self):
        request = empty_pb2.Empty()
        response = self._stub.query_running_nodes(request)
        return response.nodes

    @retry_grpc_request
    def num_nodes_waiting(self):
        request = empty_pb2.Empty()
        response = self._stub.num_nodes_waiting(request)
        return response.waiting_num

    @retry_grpc_request
    def join_rendezvous(self, node_id, local_world_size):
        request = elastic_training_pb2.NodeMeta()
        request.id = node_id
        request.local_world_size = local_world_size
        response = self._stub.join_rendezvous(request)
        return response.round

    @retry_grpc_request
    def get_comm_world(self):
        request = empty_pb2.Empty()
        response = self._stub.get_comm_world(request)
        return response.world

    @retry_grpc_request
    def report_rdzv_params(self, min_nodes, max_nodes, waiting_timeout):
        request = elastic_training_pb2.RendezvousParams()
        request.min_nodes = min_nodes
        request.max_nodes = max_nodes
        request.waiting_timeout = waiting_timeout
        response = self._stub.report_rdzv_params(request)
        return response.success

    @retry_grpc_request
    def kv_store_set(self, key, value):
        request = elastic_training_pb2.KeyValuePair()
        request.key = key
        request.value = value
        response = self._stub.kv_store_set(request)
        return response.success

    @retry_grpc_request
    def kv_store_get(self, key):
        request = elastic_training_pb2.KeyValuePair()
        request.key = key
        response = self._stub.kv_store_get(request)
        return response.value

    @retry_grpc_request
    def report_node_status(self, rank):
        if rank is None:
            return
        request = elastic_training_pb2.NodeMeta()
        request.id = self._node_id
        request.type = self._node_type
        request.rank = int(rank)
        self._stub.update_node_status(request)

    def report_failures(self, error_data):
        request = elastic_training_pb2.NodeFailure()
        request.node_id = self._node_id
        request.error_data = error_data
        self._stub.report_failure(request)


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


class LocalMasterClient(object):
    """MockedMasterClient provides the same API as MasterClient without
    any RPC call.
    """

    def __init__(self, node_id):
        """Initialize a master client.
        Args:
            worker_id: int
            the unique and ordered worker ID assigned
            by dlrover command-line.
        """
        self._node_id = node_id
        self._num_minibatches_per_shard = 0
        self._datasets: Dict[str, LocalDataset] = {}
        self._task_type = None
        self._kv_store: Dict[str, str] = {}
        self._rdzv_states: Dict[str, bytes] = {}
        self._rdzv_tokens: Dict[str, int] = {}

    def reset_dataset(self, dataset_name):
        """Reset a dataset

        Args:
            dataset_name: name of the dataset, must not be None.
        """
        dataset = self._datasets.get(dataset_name, None)
        dataset.reset()

    def get_task(self, dataset_name):
        """Get a task from master.

        Args:
            dataset_name: string
            the training phase, c.f. /dlrover/proto/dlrover.proto

        Returns:
            the task unit assigned by master,
            c.f. /dlrover/proto/dlrover.proto
        """

        shard = elastic_training_pb2.Shard()
        res = elastic_training_pb2.Task(shard=shard)
        dataset = self._datasets.get(dataset_name, None)
        if dataset:
            start, end = dataset.get_task()
            if start != -1 and end != -1:
                res.shard.start = start
                res.shard.end = end
                res.type = self._task_type
        return True, res

    def report_task_result(self, dataset_name, task_id, err_msg):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.

          exec_counters: dict
          statistics of the task being executed.
        """
        return empty_pb2.Empty()

    def update_node_event(self, task_type, task_id, enent):
        return True

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
        dataset = LocalDataset(
            batch_size,
            num_epochs,
            dataset_size,
            shuffle,
            num_minibatches_per_shard,
            task_type,
        )
        self._task_type = task_type
        self._datasets[dataset_name] = dataset

    def get_dataset_epoch(self, dataset_name):
        dataset = self._datasets.get(dataset_name, None)
        res = elastic_training_pb2.GetDatasetEpochResponse()
        if dataset:
            res.epoch = dataset.get_current_epoch()
        return res

    def report_model_metric(self, *args):
        return empty_pb2.Empty()

    def report_used_resource(self, memory, cpu):
        return empty_pb2.Empty()

    def kv_store_set(self, key, value):
        self._kv_store[key] = value
        return True

    def kv_store_get(self, key):
        return self._kv_store.get(key, "")

    def report_node_status(self, rank):
        logger.info(f"Report rank {rank}")
        return


def build_master_client(master_addr=None):
    if master_addr is None:
        master_addr = os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "")
    worker_id = int(os.getenv(NodeEnv.WORKER_ID, 0))
    worker_type = os.getenv(NodeEnv.WORKER_TYPE, "worker")

    if master_addr:
        logger.info("Build master client to connect with the DLRover master.")
        master_client = MasterClient(master_addr, worker_id, worker_type)
    else:
        master_client = LocalMasterClient(worker_id)
    return master_client


class GlobalMasterClient(object):
    MASTER_CLIENT = build_master_client()
