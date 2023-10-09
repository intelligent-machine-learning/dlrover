# Copyright 2020 The DLRover Authors. All rights reserved.
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

import pickle
import socket
import telnetlib
from contextlib import closing
from dataclasses import dataclass
from typing import Dict, List

import grpc

from dlrover.python.common.constants import GRPC
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.serialize import JsonSerializable

TIMEOUT_SEC = 5


def build_channel(addr):
    if not addr_connected(addr):
        return None
    channel = grpc.insecure_channel(
        addr,
        options=[
            ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
            ("grpc.enable_retries", True),
            (
                "grpc.service_config",
                """{ "retryPolicy":{ "maxAttempts": 5, \n
"initialBackoff": "0.2s", \n
"maxBackoff": "3s", "backoffMutiplier": 2, \n
"retryableStatusCodes": [ "UNAVAILABLE" ] } }""",
            ),
        ],
    )
    return channel


def addr_connected(addr):
    if not addr:
        return False
    host_port = addr.split(":")
    if len(host_port) != 2:
        return False
    host = host_port[0]
    port = int(host_port[1])
    try:
        telnetlib.Telnet(host=host, port=port, timeout=3)
        return True
    except socket.gaierror:
        logger.warning(f"Service {addr} is not connected.")
        return False
    except Exception as e:
        logger.error(f"Service {addr} is not connected.", e)
    return False


def find_free_port(port=0):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", port))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_free_port_in_range(start, end):
    for port in range(start, end):
        try:
            return find_free_port(port)
        except OSError as e:
            logger.info(
                f"Socket creation attempt failed with {port}.", exc_info=e
            )
    return RuntimeError(f"Fail to find a free port in [{start}, {end})")


def grpc_server_ready(channel) -> bool:
    try:
        grpc.channel_ready_future(channel).result(timeout=TIMEOUT_SEC)
        return True
    except grpc.FutureTimeoutError:
        return False


def deserialize_message(data: bytes):
    """The method will create a message instance with the content.
    Args:
        pickle_data: pickle bytes of a class instance.
    """
    message = None
    if data:
        try:
            message = pickle.loads(data)
        except Exception as e:
            logger.warning(f"Pickle failed to load {str(data)}", e)
    return message


class Message(JsonSerializable):
    def serialize(self):
        return pickle.dumps(self)


@dataclass
class TaskRequest(Message):
    dataset_name: str = ""


@dataclass
class Shard(Message):
    name: str = ""
    start: int = 0
    end: int = 0
    indices: List[int] = None  # type: ignore


@dataclass
class Task(Message):
    task_id: int = 0
    shard: Shard = Shard()
    type: int = 0
    extended_config: Dict[str, str] = None  # type: ignore


@dataclass
class GPUStats(Message):
    index: int = 0
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    gpu_utilization: float = 0


@dataclass
class TensorStats(Message):
    """TensorStats contains tensor statistics of a deep learning model"""

    variable_count: int = 0
    total_variable_size: int = 0
    max_variable_size: int = 0
    kv_embedding_dims: List[int] = None  # type: ignore


@dataclass
class OpStats(Message):
    """TensorStats contains OP statistics of a deep learning model"""

    op_count: int = 0
    update_op_count: int = 0
    read_op_count: int = 0
    input_fetch_dur: int = 0
    flops: int = 0


@dataclass
class ModelInfo(Message):
    """ModelInfo contains profiling data of a model."""

    tensor_stats: TensorStats = TensorStats()
    op_stats: OpStats = OpStats()
    instantiation_memory: int = 0
    activation_memory: int = 0


@dataclass
class ResourceStats(Message):
    memory: int = 0  # unit Byte.
    cpu: float = 0.0
    gpu_stats: List[GPUStats] = None  # type: ignore


@dataclass
class GlobalStep(Message):
    timestamp: int = 0
    step: int = 1
    elapsed_time_per_step: float = 0.0


@dataclass
class DatasetShardParams(Message):
    batch_size: int = 0
    num_epochs: int = 0
    dataset_size: int = 0
    shuffle: bool = False
    num_minibatches_per_shard: int = 0
    dataset_name: str = ""
    task_type: int = 0
    storage_type: str = ""


@dataclass
class ShardCheckpointRequest(Message):
    dataset_name: str = ""


@dataclass
class ShardCheckpoint(Message):
    content: str = ""


@dataclass
class TaskResult(Message):
    dataset_name: str = ""
    task_id: int = 0  # Task id assigned by master.
    # When error occurred, err_message contains error message in plain text.
    err_message: str = ""
    # statistics of the task being executed.
    exec_counters: Dict[str, int] = None  # type: ignore


@dataclass
class SyncJoin(Message):
    sync_name: str = ""


@dataclass
class SyncFinish(Message):
    sync_name: str = ""


@dataclass
class SyncBarrier(Message):
    barrier_name: str = ""
    notify: bool = False


@dataclass
class PsReady(Message):
    pass


@dataclass
class ClusterVersionRequest(Message):
    task_type: str = ""  # TF job type PS/worker
    task_id: int = 0
    version_type: str = ""


@dataclass
class ClusterVersion(ClusterVersionRequest):
    version: int = 0


@dataclass
class NodeMeta(Message):
    type: str = ""
    addr: str = ""
    memory: int = 0
    cpu: float = 0.0
    gpu: int = 0
    gpu_type: str = ""
    id: int = 0
    rank: int = 0
    status: str = ""


class NodeAddress(NodeMeta):
    pass


@dataclass
class NetworkStatus(NodeMeta):
    elasped_time: float = 0.0


@dataclass
class NodeEvent(Message):
    event_type: str = ""
    message: str = ""
    node: NodeMeta = NodeMeta()


@dataclass
class NodeFailure(Message):
    error_data: str = ""
    restart_count: int = 0
    level: str = ""


@dataclass
class RendezvousParams(Message):
    min_nodes: int = 0
    max_nodes: int = 0
    waiting_timeout: int = 0
    node_unit: int = 0


@dataclass
class RendezvousRequest(Message):
    node_id: int = 0
    local_world_size: int = 0
    rdzv_name: str = ""


class CommWorldRequest(RendezvousRequest):
    pass


class JoinRendezvousRequest(RendezvousRequest):
    pass


class WaitingNodeNumRequest(RendezvousRequest):
    pass


class NetworkReadyRequest(Message):
    pass


class StragglerExistRequest(Message):
    pass


@dataclass
class NetworkCheckResult(Message):
    nodes: List[int] = None  # type: ignore
    reason: str = ""


@dataclass
class RendezvousState(Message):
    world: Dict[int, int] = None  # type: ignore
    waiting_num: int = 0
    round: int = 0
    group: int = 0


@dataclass
class PsNodesRequest(Message):
    pass


@dataclass
class PsNodes(Message):
    nodes: List[NodeMeta] = None  # type: ignore
    new_ps_ready: bool = False
    ps_failure: bool = False


@dataclass
class TrainingStatusRequest(Message):
    pass


@dataclass
class TrainingStatus(Message):
    status: int = 0


@dataclass
class RunningNodesRequest(Message):
    pass


@dataclass
class RunningNodes(Message):
    nodes: List[NodeMeta] = None  # type: ignore


@dataclass
class KeyValuePair(Message):
    key: str = ""
    value: bytes = b""


@dataclass
class DataLoaderConfig(Message):
    """The configured parameters of DataLoader.
    Attr:
        dataloader_name: a DataLoader instance has an unique name in a job.
        batch_size: the number of samples in a batch.
        num_workers: how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process.
        pin_memory: If True, the data loader will copy Tensors into
            device/CUDA pinned memory before returning them.
    """

    version: int = 0
    dataloader_name: str = ""
    last_batch_size: int = 0
    batch_size: int = 0
    num_workers: int = 0
    pin_memory: int = 0


@dataclass
class OptimizerConfig(Message):
    version: int = 0
    optimizer_name: str = ""
    learning_rate: float = 0.0
    weight_decay: float = 0.0


@dataclass
class ParallelConfigRequest(Message):
    pass


@dataclass
class ParallelConfig(Message):
    dataloader: DataLoaderConfig = DataLoaderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
