# Copyright 2023 The DLRover Authors. All rights reserved.
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
import functools
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import psutil
import torch
import torch.distributed.elastic.timer as timer
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import (
    DEFAULT_ROLE,
    RunResult,
    Worker,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
    _get_fq_hostname,
    _RoleInstanceInfo,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
)
from torch.distributed.elastic.metrics import put_metric
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import (
    PContext,
    SignalException,
    Std,
)
from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
)
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.api import RendezvousHandler
from torch.distributed.launcher.api import LaunchConfig, _get_entrypoint_name

from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    Accelerators,
    AscendConstants,
    ConfigPath,
    JobConstant,
    NodeEnv,
    NodeErrorMessage,
    NodeEventType,
    RendezvousName,
    TrainingExceptionLevel,
)
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisActionType
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
    NodeAction,
)
from dlrover.python.elastic_agent.config.paral_config_tuner import (
    ParalConfigTuner,
)
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.diagnosis.diagnosis_agent import (
    DiagnosisAgent,
)
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.elastic_agent.monitor.training import TorchTrainingMonitor
from dlrover.python.elastic_agent.torch.ckpt_saver import AsyncCheckpointSaver
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore
from dlrover.python.util.common_util import (
    find_free_port_for_hccl,
    find_free_port_in_range,
    find_free_port_in_set,
)
from dlrover.python.util.numa_util import get_gpu_affinity, get_npu_affinity
from dlrover.python.util.time_util import timestamp_diff_in_seconds
from dlrover.trainer.torch.utils import (
    version_less_than_230,
    version_less_than_240,
)

try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except (ModuleNotFoundError, ImportError):  # noqa: F841
    pass

__all__ = ["launch_agent"]


def _set_paral_config():
    """
    Set up the directory and path for the parallelism configuration.
    """
    config_dir = os.path.dirname(ConfigPath.PARAL_CONFIG)
    os.makedirs(config_dir, exist_ok=True)
    os.environ[ConfigPath.ENV_PARAL_CONFIG] = ConfigPath.PARAL_CONFIG
    os.environ[ConfigPath.ENV_RUNTIME_METRICS] = ConfigPath.RUNTIME_METRICS


def _get_local_ip():
    local_ip = os.getenv("POD_IP", "")
    if not local_ip:
        try:
            local_ip = socket.gethostbyname(_get_fq_hostname())
        except socket.gaierror:
            logger.warning(
                "Can not resolve host IP. " "Use default '127.0.0.1' instead."
            )
            local_ip = "127.0.0.1"
    return local_ip


class RendezvousOutSyncError(Exception):
    pass


class NodeCheckFailedError(RuntimeError):
    pass


@dataclass
class ElasticLaunchConfig(LaunchConfig):
    """
    Creates a rendezvous config of elastic training.

    Args:
        precheck: the level to run pre-check task before starting
            the training task.
        network_check: whether to check the network available before training.
        comm_perf_test: whether to test the communication performance.
        node_unit: the number of unit of nodes. The number of nodes must be
            a multiple of node_unit.
        auto_config: indicate if automatically configure the nnodes and
            nproc_per_node.
        auto_tunning: whether to auto-tune the parallelism configuration.
        exclude_straggler: The node will exit if it is a straggler in network
            check and exclude_straggler is True.
        save_at_breakpoint: indicate if save the checkpoint from the shared
            memory into the disk after a failure occurs.
        accelerator: the type of accelerator processor like nvidia.com/gpu,
            ascend-npu.
        training_log_file: the training log file of this training job
        failure_node_errors: the error information that indicate the node
            is a failure node
    """

    precheck: int = 0
    network_check: bool = False
    comm_perf_test: bool = False
    node_unit: int = 1
    training_port: int = AscendConstants.HCCL_PORT_START_DEFAULT
    auto_config: bool = False
    auto_tunning: bool = False
    exclude_straggler: bool = False
    save_at_breakpoint: bool = False
    accelerator: str = ""
    log_dir: Optional[str] = None  # Keep Compatibility with PyTorch>=2.3.0
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    training_log_file: str = ""
    failure_node_errors: str = ""
    numa_affinity: bool = False

    def set_node_unit(self, node_unit):
        """Set the number unit of nodes."""
        self.node_unit = node_unit
        self.rdzv_configs["node_unit"] = node_unit

    def auto_configure_params(self):
        self.training_log_file = os.getenv(NodeEnv.TRAINING_LOG_FILE, "")
        self.failure_node_errors = os.getenv(NodeEnv.FAILURE_NODE_ERRORS, "")
        if len(self.failure_node_errors) > 0:
            errors = self.failure_node_errors.strip()
            if errors[0] != "#" or errors[-1] != "#":
                logger.warning("invalid failure node errors: %s", errors)
                self.failure_node_errors = ""

        device = ""
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name()
        if "Ascend" in device:
            self.accelerator = Accelerators.ASCEND_NPU
        logger.info(
            f"Use {self.accelerator} device for training, "
            f"cuda is available: {torch.cuda.is_available()}."
        )

        if not self.auto_config:
            return

        if NodeEnv.NODE_NUM in os.environ:
            self.min_nodes = int(os.environ[NodeEnv.NODE_NUM])
            self.max_nodes = int(os.environ[NodeEnv.NODE_NUM])
        if torch.cuda.is_available():
            self.nproc_per_node = torch.cuda.device_count()
        if self.min_nodes >= 4:
            self.network_check = True

    def update_precheck_args(self):
        if self.precheck == 0:
            self.comm_perf_test = False or self.comm_perf_test
            self.network_check = False or self.network_check

        if self.precheck == 1:
            self.network_check = True
            self.comm_perf_test = False or self.comm_perf_test

        if self.precheck == 2:
            self.network_check = True
            self.comm_perf_test = True


class MasterRendezvousHandler(RendezvousHandler):
    """The rendezvous handler completes rendezvous by connecting
    with the ElasticJob master. The master will collect all nodes
    after the handler of all node agents calls `_join_rendezvous`.
    Then, the handler will get the communication world from the master
    and assign ranks to the training process.

    Args:
        name: the name of rendezvous.
        node_rank: the node rank.
        rdzv_params: RendezvousParameters instance. We can set timeout of
            rendezvous in the rdzv_params.config. Now we set:
            join_timeout: the timeout to join the rendezvous. The timeout
                happens if the number of nodes is less than min_nodes
                in the join_timeout.
            lastcall_timeout: the timeout to wait new nodes after the
                number of nodes is equal or greater than min_nodes.
                The node will join the rendezvous to start train if
                the timeout happens.
            pend_timeout: the timeout to wait the next rendezvous. The timeout
                happens if there is a rendezvous and the node is not in the
                rendzvous. For example. the number of nodes must be the
                multiple of node_uint. If the node_uint = 4 and the number
                of nodes is 5, then the 5th node will wait for more nodes
                in the pend_timeout.
            local_world_size: the number of local processes.
    """

    def __init__(
        self,
        name,
        node_rank,
        rdzv_params: RendezvousParameters,
        local_world_size,
    ):
        self._name = name
        self._node_rank = node_rank
        self._rdzv_params = rdzv_params
        self._local_world_size = local_world_size
        self.join_timeout = int(
            rdzv_params.get(
                "join_timeout", JobConstant.RDZV_JOIN_TIMEOUT_DEFAULT
            )
        )
        self.pend_timeout = float(rdzv_params.get("pend_timeout", "inf"))
        self._client = MasterClient.singleton_instance()
        self._store = MasterKVStore(self._name, timedelta(seconds=60))
        lastcall_timeout = int(rdzv_params.get("lastcall_timeout", 60))
        node_unit = int(rdzv_params.get("node_unit", "1"))
        self._client.report_rdzv_params(
            rdzv_params.min_nodes,
            rdzv_params.max_nodes,
            lastcall_timeout,
            node_unit,
            self.join_timeout,
        )

    def get_backend(self) -> str:
        return "dlrover-master"

    def is_closed(self) -> bool:
        return False

    def set_closed(self):
        """Marks the rendezvous as closed."""
        pass

    def _join_rendezvous(self):
        """The node join a rendezvous by sending its
        ID and local world size.
        """
        round = self._client.join_rendezvous(
            self._node_rank, self._local_world_size, rdzv_name=self._name
        )
        return round

    def next_rendezvous(self):
        """The handler will periodically query the world from the master until
        the world is not empty. The world is a dictionary like
        {0: 8, 1: 8, 2: 8} where the key is the node ID and the value is
        the local world size. The handler can get its rank by the position
        of it node ID in the world.
        """
        start_join = time.time()
        node_name = os.getenv("POD_NAME", "")
        msg = (
            f"The node {node_name} with rank {self._node_rank} attempts to "
            f"join the next round of the rendezvous {self._name} "
            f"with timeout {self.join_timeout}."
        )
        logger.info(msg)
        self._join_rendezvous()

        start_pending = 0
        while True:
            self._check_network_rdzv_for_elastic_training()
            round, group, world = self._client.get_comm_world(
                self._name, self._node_rank
            )
            if world:
                if self._node_rank in world:
                    break
                else:
                    if start_pending == 0:
                        logger.info(
                            "The node is not in the world "
                            "and waits for more nodes."
                        )
                        start_pending = time.time()
                    time.sleep(JobConstant.RENDEZVOUS_DEFAULT_INTERVAL)
                    start_join = time.time()
                    if start_join - start_pending > self.pend_timeout:
                        raise TimeoutError(
                            f"Timeout {self.pend_timeout}s to wait more nodes"
                        )
                    continue
            elif time.time() - start_join > self.join_timeout:
                timeout = self.join_timeout
                err_msg = (
                    f"Timeout {timeout}s to wait the enough nodes "
                    "to complete rendezvous."
                )
                self._report_failure(
                    err_msg, level=TrainingExceptionLevel.RDZV_ERROR
                )
                raise TimeoutError(err_msg)
            time.sleep(JobConstant.RENDEZVOUS_DEFAULT_INTERVAL)
        rank = list(world.keys()).index(self._node_rank)
        world_size = len(world)
        logger.info(
            f"The node {node_name} has joined round {round} of "
            f"the {self._name} rendezvous as rank {rank} in a world of size "
            f"{world_size}."
        )
        if (
            self._name == RendezvousName.ELASTIC_TRAINING
            and world_size < self._rdzv_params.max_nodes
        ):
            err_msg = f"Scale down the number of nodes to {world_size}"
            self._report_failure(err_msg, level=TrainingExceptionLevel.WARNING)
        store = self._get_store(round, group)
        return store, world

    def _check_network_rdzv_for_elastic_training(self):
        """The worker need to exit the elastic-training rendezvous if there are
        workers to join the network-check rendezvous.
        """
        if self._name == RendezvousName.ELASTIC_TRAINING:
            num = self._client.num_nodes_waiting(RendezvousName.NETWORK_CHECK)
            if num > 0:
                raise RendezvousOutSyncError(
                    "Some workers join the network-check rendezvous"
                    "not the elastic-training rendezvous."
                )

    def _report_failure(self, err_msg, level):
        if self._node_rank == 0:
            self._client.report_failures(err_msg, 0, level)

    def _get_store(self, round, group) -> Store:
        key_prefix = f"torch.rendezvous.{self._name}.{round}.{group}"
        return PrefixStore(key_prefix, self._store)

    def num_nodes_waiting(self) -> int:
        return self._client.num_nodes_waiting(self._name)

    def get_run_id(self) -> str:
        """Returns the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """
        return self._rdzv_params.run_id

    def shutdown(self) -> bool:
        """Closes all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """
        pass


class ElasticTrainingAgent(LocalElasticAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers.
    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The agent select to fail or relaunch subprocesses according to the
    failed reason of subprocess. Now, if the exitcode is not 1, the agent
    will fail and the DLRover will relaunch the node. Because, we find
    the exitcode is 1 if the hardware breakdowns.
    """

    def __init__(
        self,
        node_rank,
        config: ElasticLaunchConfig,
        entrypoint,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
        training_log_file: str = "",
        failure_node_errors: str = "",
        with_diagnostician: bool = True,
    ):
        if version_less_than_230():
            super().__init__(
                spec=spec,
                exit_barrier_timeout=exit_barrier_timeout,
            )
        else:
            super().__init__(
                spec=spec,
                logs_specs=config.logs_specs,
                exit_barrier_timeout=exit_barrier_timeout,
            )
        self._node_rank = node_rank
        self._config = config
        self._entrypoint = entrypoint
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        self._worker_watchdog: Optional[timer.FileTimerServer] = None
        self._restart_count = 0
        self._remaining_failovers = self._remaining_restarts
        self._client = MasterClient.singleton_instance()
        if config.auto_tunning:
            self._paral_config_tuner = ParalConfigTuner.singleton_instance()
            self._paral_config_tuner.start()

        self._save_ckpt_executor = ThreadPoolExecutor(max_workers=1)
        self._save_ckpt_future = None

        if with_diagnostician:
            self._diagnose_agent = DiagnosisAgent.singleton_instance(
                training_log_file, failure_node_errors, node_rank
            )
        self._agent_context = get_agent_context()
        self._rank_cpu_affinity = {}
        if self._config.numa_affinity:
            for rank in range(self._config.nproc_per_node):
                if self._config.accelerator == Accelerators.ASCEND_NPU:
                    self._rank_cpu_affinity[rank] = get_npu_affinity(rank)
                else:
                    self._rank_cpu_affinity[rank] = get_gpu_affinity(rank)
                logger.info(
                    f"get rank {rank} affinity: "
                    f"{self._rank_cpu_affinity[rank]}"
                )

    @prof
    def _stop_workers_ascend(self, worker_group: WorkerGroup) -> None:
        """The ASCEND framework might fork multiple sub-processes, we should
        stop all the children processes before shutdown the workers.
        """

        logger.info("stop workers via SIGKILL for Ascend NPU")
        # print out a snapshot of all processes
        env_utils.print_process_list()

        if self._pcontext is not None:
            pc_pids = set(self._pcontext.pids().values())
            logger.info(f"try to kill child processes of {pc_pids}")
            for pid in pc_pids:
                try:
                    pp = psutil.Process(pid)
                    cp = pp.children()
                    for proc in cp:
                        logger.info(f"kill sub {proc.pid} of parent {pid}")
                        os.kill(proc.pid, signal.SIGKILL)
                except Exception as e:
                    logger.warning(f"error when kill {pid}: {str(e)}")

        self._shutdown(death_sig=signal.SIGKILL)

        # cleanup orphan processes if exists
        self._stop_orphan_workers(worker_group)

        # print out a snapshot of all processes again
        env_utils.print_process_list()

    @prof
    def _stop_orphan_workers(self, wg: WorkerGroup) -> None:
        """How we define the orphan workers
        1. ppid == 1
        2. is_worker_process() is True
        """
        try:
            for p in psutil.process_iter():
                if p.ppid() == 1 and env_utils.is_worker_process(p.pid):
                    name = " ".join(p.cmdline())
                    logger.info(f"find orphan workers {p.pid}: {name}")
                    os.kill(p.pid, signal.SIGKILL)
        except Exception as e:
            logger.warning(f"_stop_orphan_workers exception: {e}")

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""
        Runs rendezvous for the workers specified by worker spec.
        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """

        spec = worker_group.spec
        store, world = spec.rdzv_handler.next_rendezvous()
        self._store = store
        group_world_size = len(world)
        group_rank = list(world.keys()).index(self._node_rank)

        workers = self._assign_worker_ranks(self._node_rank, world, spec)
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size

        if group_rank == 0:
            spec.master_port = self._get_free_port()

            if hasattr(spec, "local_addr"):
                self._set_master_addr_port(
                    store,
                    spec.master_addr,
                    spec.master_port,
                    spec.local_addr,
                )
            else:
                # Compatible with torch 1.x
                self._set_master_addr_port(
                    store,
                    spec.master_addr,
                    spec.master_port,
                )

        master_addr, master_port = self._safe_get_master_addr_port(store)

        # compatible with torch 2.4
        if not version_less_than_240():
            worker_group.master_addr = master_addr
            worker_group.master_port = master_port

        logger.info(
            f"[{spec.role}] Rendezvous complete for workers. Result:\n"
            f"  restart_count={self._restart_count}\n"
            f"  master_addr={master_addr}\n"
            f"  master_port={master_port}\n"
            f"  group_rank={group_rank}\n"
            f"  group_world_size={group_world_size}\n"
            f"  local_ranks={[worker.local_rank for worker in workers]}\n"
            f"  role_ranks={[worker.role_rank for worker in workers]}\n"
            f"  global_ranks={[worker.global_rank for worker in workers]}\n"
            f"  role_world_sizes="
            f"{[worker.role_world_size for worker in workers]}\n"
            f"  global_world_sizes="
            f"{[worker.world_size for worker in workers]}\n"
        )

    """
    The following function(copied from torch 230) is used to
    compatible with torch < 240
    """

    def _set_master_addr_port(
        self,
        store: Store,
        master_addr: Optional[str],
        master_port: Optional[int],
        local_addr: Optional[str] = None,
    ):
        if master_port is None:
            sock = self._get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

        if master_addr is None:
            # If user specified the address for the local node,
            # use it as the master addr if not exist
            if local_addr:
                master_addr = local_addr
            else:
                master_addr = _get_fq_hostname()

        store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

    def _get_master_addr_port(self, store: Store) -> Tuple[str, int]:
        master_addr = store.get("MASTER_ADDR").decode(encoding="UTF-8")
        master_port = int(store.get("MASTER_PORT").decode(encoding="UTF-8"))
        return master_addr, master_port

    def _safe_get_master_addr_port(self, store: Store) -> Tuple[str, int]:
        for _ in range(5):
            try:
                return self._get_master_addr_port(store)
            except Exception as e:
                logger.warning(
                    f"_get_master_addr_port failed with exception {e}"
                )
                time.sleep(10)

        raise ValueError("invalid value in _get_master_addr_port")

    def _get_socket_with_port(self) -> socket.socket:
        """Return a free port on localhost.

        The free port is "reserved" by binding a temporary socket on it.
        Close the socket before passing the port to the entity that
        requires it. Usage example::

        sock = _get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]
            sock.close()
            # there is still a race-condition that some other process
            # may grab this port before func() runs
            func(port)
        """
        addrs = socket.getaddrinfo(
            host="localhost",
            port=None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        for addr in addrs:
            family, type, proto, _, _ = addr
            s = socket.socket(family, type, proto)
            try:
                s.bind(("localhost", 0))
                s.listen(0)
                return s
            except OSError as e:
                s.close()
                logger.info("Socket creation attempt failed.", exc_info=e)
        raise RuntimeError("Failed to create a socket")

    """
    The above function(copied from torch 230) is used to
    compatible with torch < 240
    """

    def _get_free_port(self):
        """Find a free port from the HOST_PORTS in env."""
        free_port = None
        host_ports = os.getenv("HOST_PORTS", "")
        if host_ports:
            ports = []
            for port in host_ports.split(","):
                ports.append(int(port))
            try:
                free_port = find_free_port_in_set(ports)
            except RuntimeError as e:
                logger.warning(e)
        if not free_port:
            free_port = find_free_port_in_range(20000, 30000)
        return free_port

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _assign_worker_ranks(
        self, node_id, world, spec: WorkerSpec
    ) -> List[Worker]:
        """
        Determines proper ranks for worker processes. The rank assignment
        is done according to the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. Each agent retrieves configuration for all agents
           and performs two level sort using role and rank.
        3. Determine the global rank: the global rank of workers for the
           current agent is the offset of  infos array up to group_rank
           of the agent. The offset is computed as a sum of local_world_size
           of all agents that have rank less than the group_rank.
           The workers would have the ranks: [offset, offset+local_world_size)
        4. Determine the role rank: The role rank is determined using the
           algorithms in the point 3 with the exception that the offset is
           done from the first agent that has the same role as current one
           and has the minimum group rank.
        """

        role_infos: List[_RoleInstanceInfo] = []
        nodes = list(world.keys())
        for i, local_world_size in world.items():
            group_rank = nodes.index(i)
            role_info = _RoleInstanceInfo(
                spec.role, group_rank, local_world_size
            )
            role_infos.append(role_info)
        group_rank = nodes.index(node_id)

        if version_less_than_240():
            my_role_info = role_infos[group_rank]
            worker_world_size, worker_global_ranks = self._get_ranks(
                role_infos, group_rank
            )
            role_infos = sorted(
                role_infos, key=functools.cmp_to_key(_RoleInstanceInfo.compare)
            )
            (
                role_start_idx,
                role_end_idx,
            ) = _RoleInstanceInfo.find_role_boundaries(
                role_infos, my_role_info.role
            )
            role_pos = next(
                idx
                for idx, role_info in enumerate(role_infos)
                if _RoleInstanceInfo.compare(role_info, my_role_info) == 0
            )
            role_world_size, role_ranks = self._get_ranks(
                role_infos, role_pos, role_start_idx, role_end_idx + 1
            )
            workers = []
            for ind in range(spec.local_world_size):
                worker = Worker(
                    local_rank=ind,
                    global_rank=worker_global_ranks[ind],
                    role_rank=role_ranks[ind],
                    world_size=worker_world_size,
                    role_world_size=role_world_size,
                )
                workers.append(worker)
        else:
            group_world_size = len(world)

            ROLE_INFO_PREFIX = "torchelastic/role_info/"
            ASSIGNED_RANKS_PREFIX = "torchelastic/assigned_ranks/"

            agent_role_info = _RoleInstanceInfo(
                spec.role, group_rank, spec.local_world_size
            )
            self._store.set(
                f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize()
            )

            if group_rank == 0:
                role_infos_bytes = self._store.multi_get(
                    [
                        f"torchelastic/role_info/{i}"
                        for i in range(group_world_size)
                    ]
                )
                role_infos = [
                    _RoleInstanceInfo.deserialize(info_bytes)
                    for info_bytes in role_infos_bytes
                ]

                role_sizes: DefaultDict[str, int] = defaultdict(lambda: 0)
                global_size = 0
                for role_info in role_infos:
                    role_sizes[role_info.role] += role_info.local_world_size
                    global_size += role_info.local_world_size

                base_global_rank = 0
                role_ranks = defaultdict(lambda: 0)

                keys = []
                values = []
                for i, role_info in enumerate(role_infos):
                    keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
                    values.append(
                        json.dumps(
                            [
                                base_global_rank,
                                global_size,
                                role_ranks[role_info.role],
                                role_sizes[role_info.role],
                            ]
                        )
                    )

                    base_global_rank += role_info.local_world_size
                    role_ranks[role_info.role] += role_info.local_world_size

                self._store.multi_set(keys, values)

            # get will block until the data is available in the store.
            (
                base_global_rank,
                global_world_size,
                base_role_rank,
                role_world_size,
            ) = json.loads(
                self._store.get(f"{ASSIGNED_RANKS_PREFIX}{group_rank}")
            )

            workers = []
            for local_rank in range(spec.local_world_size):
                worker = Worker(
                    local_rank=local_rank,
                    global_rank=base_global_rank + local_rank,
                    role_rank=base_role_rank + local_rank,
                    world_size=global_world_size,
                    role_world_size=role_world_size,
                )
                workers.append(worker)
        return workers

    def _initialize_workers(self, worker_group, max_errors=3):
        logger.info(
            "Start initializing "
            f"training({self.__class__.__name__}) workers."
        )
        start_pending = 0
        err_cnt = 0
        pend_timeout = float(
            self._config.rdzv_configs.get("pend_timeout", "inf")
        )
        while True:
            try:
                if self._config.network_check:
                    run_network_check(self._config, self._entrypoint)
                super()._initialize_workers(worker_group)
                # We need to register handler after starting workers because
                # the PContext start_worker will overwrite the handler.
                AsyncCheckpointSaver.register_signal_handler()
            except RendezvousOutSyncError:
                if start_pending == 0:
                    start_pending = time.time()
                    logger.info(
                        "Exit elastic-training rendezvous when there are "
                        "agents to join the network-check rendezvous."
                    )
                time.sleep(JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL)
                if time.time() - start_pending > pend_timeout:
                    raise TimeoutError("Timeout to wait for new nodes.")
            except NodeCheckFailedError as node_check_error:
                raise node_check_error
            except Exception as e:
                err_cnt += 1
                if err_cnt < max_errors:
                    stack_trace = traceback.format_exc()
                    logger.error(
                        f"Unexpected exception in _initialize_workers: {e}\n"
                        f"Stack backtrace:\n {stack_trace}"
                    )
                    self._stop_workers(worker_group)
                    continue
                else:
                    raise e
            else:
                logger.info("Finish initializing training workers.")
                break

    @prof
    def _stop_workers(
        self, worker_group: WorkerGroup, is_restart=False, timeout=300
    ) -> None:
        try:
            signal.signal(signal.SIGALRM, self._stop_timeout_handler)
            signal.alarm(timeout)

            if self._config.accelerator == Accelerators.ASCEND_NPU:
                self._stop_workers_ascend(worker_group)
            else:
                if version_less_than_240():
                    super()._stop_workers(worker_group)
                else:
                    super()._stop_workers(worker_group, is_restart)

            signal.alarm(0)
        except TimeoutError as te:
            logger.error(str(te))
            raise
        finally:
            signal.alarm(0)

    def _stop_timeout_handler(self, signum, frame):
        raise TimeoutError("Timed out waiting for stopping workers.")

    def _set_numa_affinity(self):
        """set numa affinity to workers processes,
        as well as its children processes
        """
        for local_rank, pid in self._pcontext.pids().items():
            if self._rank_cpu_affinity[local_rank] is not None:
                try:
                    os.sched_setaffinity(
                        pid, self._rank_cpu_affinity[local_rank]
                    )
                    logger.info(
                        f"set rank {local_rank} worker {pid} affinity: "
                        f"{self._rank_cpu_affinity[local_rank]}"
                    )
                    pp = psutil.Process(pid)
                    cp = pp.children(recursive=True)

                    for p in cp:
                        os.sched_setaffinity(
                            p, self._rank_cpu_affinity[local_rank]
                        )
                        logger.info(
                            f"set rank {local_rank} child {p} affinity: "
                            f"{self._rank_cpu_affinity[local_rank]}"
                        )
                except Exception as e:
                    logger.warning(
                        f"set rank {local_rank} affinity failed: {e} "
                        f"{self._rank_cpu_affinity[local_rank]}"
                    )
            else:
                logger.warning(
                    f"rank {local_rank} worker {pid} invalid affinity"
                )

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # sync hccl port for NPU
        self.sync_training_ports()

        # Start a thread to save the checkpointing state dict from
        # the shared memory to the storage.
        AsyncCheckpointSaver.start_async_saving_ckpt()

        spec = self._worker_group.spec
        role = spec.role

        # TODO: call master to get approval of
        #  training starting(to wait pre-check)

        logger.info(
            f"[{role}] starting training workers for entrypoint: "
            f"{spec.get_entrypoint_name()}"
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        # set workers numa-affinity if necessary
        if self._config.numa_affinity:
            self._set_numa_affinity()

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)

            self._check_and_process_diagnosis_action()
            try:
                run_result: RunResult = self._monitor_workers(
                    self._worker_group
                )
            except json.decoder.JSONDecodeError:
                run_result = RunResult(state=WorkerState.FAILED)
            state = run_result.state
            self._worker_group.state = state

            put_metric(
                f"workers.{role}.remaining_restarts", self._remaining_failovers
            )
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                logger.info(
                    f"[{role}] worker group successfully finished."
                    f" Waiting {self._exit_barrier_timeout} seconds "
                    "for other agents to finish."
                )

                try:
                    self._exit_barrier()
                    logger.info("Barrier exited.")

                    self._wait_async_saver()
                    logger.info("Async saver stopped.")
                except Exception as e:
                    logger.warning(f"Unexpected exception when ending: {e}")
                finally:
                    self._client.report_succeeded_exited()
                    logger.info("Succeeded and exit.")

                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                logger.error(f"The worker fails with {run_result.failures}")
                self._save_ckpt_to_storage()

                self._agent_context.update_context(
                    worker_spec=self._worker_group.spec,
                    remaining_failovers=self._remaining_failovers,
                    restart_count=self._restart_count,
                    run_result=run_result,
                )
                try:
                    action = self._diagnose_agent.diagnose_training_failure()
                except Exception as e:
                    logger.warning(f"Failed to diagnose errors: {e}")
                    if self._remaining_failovers > 0:
                        action = NodeAction(
                            action_type=DiagnosisActionType.RESTART_WORKER,
                        )
                    else:
                        action = NodeAction(
                            action_type=DiagnosisActionType.RELAUNCH_WORKER,
                        )
                self._process_diagnosis_action(action)
                if self._worker_group.state == WorkerState.FAILED:
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                if self._membership_changed(role, rdzv_handler):
                    self._save_ckpt_to_storage()
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] worker group in {state.name} state")

    def _process_diagnosis_action(self, action: DiagnosisAction):
        if isinstance(action, NodeAction):
            action.__class__ = NodeAction
            if action.action_type == DiagnosisActionType.RESTART_WORKER:
                logger.info(
                    f"exec diagnosis action: "
                    f"{action.action_type} {action.instance}"
                )
                self._remaining_failovers -= 1
                self._restart_workers(self._worker_group)
            elif action.action_type == DiagnosisActionType.RELAUNCH_WORKER:
                logger.info(
                    f"exec diagnosis action: "
                    f"{action.action_type} {action.instance}"
                )
                self._stop_workers(self._worker_group)
                self._worker_group.state = WorkerState.FAILED
        elif isinstance(action, EventAction):
            action.__class__ = EventAction
            labels = action.event_labels
            if labels is None:
                labels = {}
            self._client.report_event(
                event_type=action.event_type,
                instance=action.event_instance,
                action=action.event_action,
                msg=action.event_msg,
                labels=labels,
            )

    def _check_and_process_diagnosis_action(self):
        action = self._agent_context.next_diagnosis_action()
        if isinstance(action, NoAction):
            return
        self._process_diagnosis_action(action)
        # avoid to execute the same event action too frequently
        if isinstance(action, EventAction) and not action.is_expired():
            time_diff = timestamp_diff_in_seconds(
                action.timestamp, datetime.now().timestamp()
            )
            expired_time_period = action.expired_time_period - time_diff
            if expired_time_period < 0:
                expired_time_period = 0
            action.update_timestamp(
                timestamp=datetime.now().timestamp(),
                expired_time_period=expired_time_period,
                executable_time_period=expired_time_period + 60,
            )
            self._agent_context.enqueue_diagnosis_action(action)

    def _wait_async_saver(self):
        """
        The agent waits for saving the checkpoint from the shared memory
        before exiting.
        """
        saver = AsyncCheckpointSaver.get_ckpt_saver()
        if saver:
            # Wait the saver finishes writing the checkpoint from the shared
            # memory to the storage.
            start_wait_time = time.time()
            while saver.wait_saving_checkpoint():
                time.sleep(JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL)
                wait_time = round(time.time() - start_wait_time, 2)
                logger.info(
                    "Wait for saving the checkpoint and "
                    f"the waiting time is {wait_time}s."
                )

    def _save_ckpt_to_storage(self):
        """
        The agent can save the checkpointing state dict in the shared
        memory into the storage before restarting training processes.
        """
        saver: AsyncCheckpointSaver = AsyncCheckpointSaver.get_ckpt_saver()
        if saver and self._config.save_at_breakpoint:
            logger.info("Start saving checkpoint at the breakpoint.")
            self._save_ckpt_future = self._save_ckpt_executor.submit(
                saver.save_shm_to_storage, 60, self._client
            )

    def _stop_workers_to_restart(self):
        """
        The agent query from the dlrover job master to check whether to restart
        workers. If true, the agent firstly stops all workers.
        """
        restart = self._client.need_to_restart_training()
        if not restart:
            return
        self._stop_workers(self._worker_group)

    def _report_failure_to_master(self, failures: Dict[int, ProcessFailure]):
        errors = {}
        if len(failures) == 0:
            return
        for rank, failure in failures.items():
            dt = str(datetime.fromtimestamp(int(failure.timestamp)))
            error = ProcessError(
                failure.local_rank, failure.exitcode, failure.message, dt
            )
            errors[rank] = error.__dict__
        error_data = json.dumps(errors)
        self._client.report_failures(
            error_data,
            self._restart_count,
            TrainingExceptionLevel.PROCESS_ERROR,
        )

    def _restart_workers(self, worker_group: WorkerGroup):
        self._restart_count += 1
        self._remaining_restarts -= 1
        # Release the shared memory lock before starting workers.
        AsyncCheckpointSaver.reset()
        super()._restart_workers(worker_group)

    def _membership_changed(self, role, rdzv_handler: RendezvousHandler):
        # Timeout may happen when to query TCPStore.
        if self._config.network_check:
            num_nodes_waiting = self._client.num_nodes_waiting(
                RendezvousName.NETWORK_CHECK
            )
        else:
            num_nodes_waiting = rdzv_handler.num_nodes_waiting()
        group_rank = self._worker_group.group_rank
        if num_nodes_waiting > 0:
            logger.info(
                f"[{role}] Detected {num_nodes_waiting} "
                f"new nodes from group_rank={group_rank}; "
                f"will restart worker group"
            )
            return True
        return False

    def stop_executor(self):
        """Shutdown the executor to save the checkpoint."""
        self._save_ckpt_executor.shutdown(wait=False)

    def sync_training_ports(
        self, interval=JobConstant.SYNC_PORTS_DEFAULT_INTERVAL
    ):
        logger.info(f"Accelerator: {self._config.accelerator}")
        if (
            self._config.accelerator == Accelerators.ASCEND_NPU
            and self._config.training_port > 0
        ):
            default_port_from_env = env_utils.get_env(
                AscendConstants.HCCL_PORT_START
            )
            # use default port from env
            if default_port_from_env:
                start_port = int(default_port_from_env)
            else:
                start_port = self._config.training_port

            port = 0
            logger.info("synchronize worker training ports...")
            count = 0
            max_count = 120
            while True:
                if count >= max_count:
                    logger.error(
                        f"exhausted {max_count} sync time. use default port"
                    )
                    break
                time.sleep(interval)
                count = count + 1
                if port == 0:
                    port = find_free_port_for_hccl(start_port)
                if port == 0:
                    logger.error(
                        f"fail to find available ports from {start_port}"
                    )
                    break
                resp = self._client.sync_training_ports(port)
                if not resp:
                    continue
                if resp.port > 0:
                    logger.info(f"config hccl port: {resp.port}")
                    os.environ[AscendConstants.HCCL_PORT_START] = str(
                        resp.port
                    )
                    break
                elif resp.newport > 0:
                    start_port = resp.newport
                    port = 0


def launch_agent(
    config: ElasticLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> Dict[int, Any]:
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning(
            f"config has no run_id, generated a random run_id: {run_id}"
        )
        config.run_id = run_id

    entrypoint_name = _get_entrypoint_name(entrypoint, args)
    node_rank = env_utils.get_node_rank()

    logger.info(
        f"Starting training agent with launch configs:\n"
        f"  entrypoint       : {entrypoint_name}\n"
        f"  min_nodes        : {config.min_nodes}\n"
        f"  max_nodes        : {config.max_nodes}\n"
        f"  nproc_per_node   : {config.nproc_per_node}\n"
        f"  run_id           : {config.run_id}\n"
        f"  rdzv_backend     : {config.rdzv_backend}\n"
        f"  rdzv_endpoint    : {config.rdzv_endpoint}\n"
        f"  rdzv_configs     : {config.rdzv_configs}\n"
        f"  max_restarts     : {config.max_restarts}\n"
        f"  monitor_interval : {config.monitor_interval}\n"
        f"  log_dir          : {config.log_dir}\n"
        f"  metrics_cfg      : {config.metrics_cfg}\n"
        f"  training_log     : {config.training_log_file}\n"
        f"  failure_errors   : {config.failure_node_errors}\n"
        f"  numa_affinity    : {config.numa_affinity}\n"
        f"  accelerator      : {config.accelerator}\n"
    )

    _set_paral_config()
    monitor = TorchTrainingMonitor(ConfigPath.RUNTIME_METRICS)
    monitor.start()

    spec = _create_worker_spec(
        node_rank=node_rank,
        rdzv_name=RendezvousName.ELASTIC_TRAINING,
        config=config,
        entrypoint=entrypoint,
        args=args,
    )
    agent = ElasticTrainingAgent(
        node_rank=node_rank,
        config=config,
        entrypoint=entrypoint,
        spec=spec,
        start_method=config.start_method,
        log_dir=config.log_dir,
        training_log_file=config.training_log_file,
        failure_node_errors=config.failure_node_errors,
    )

    shutdown_rdzv = True
    is_node_check_failed = False
    result = None

    try:
        metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))

        result = agent.run()
        # records that agent.run() has succeeded NOT
        # that workers have succeeded
        events.record(agent.get_event_succeeded())

        if result.is_failed():
            # ChildFailedError is treated specially by @record
            # if the error files for the failed children exist
            # @record will copy the first error (root cause)
            # to the error file of the launcher process.
            raise ChildFailedError(
                name=entrypoint_name,
                failures=result.failures,
            )

        return result.return_values
    except ChildFailedError:
        raise
    except SignalException:
        # when the agent dies with a signal do NOT shutdown the rdzv_handler
        # since this closes the rendezvous on this rdzv_id permanently and
        # prevents any additional scaling events
        shutdown_rdzv = False
        events.record(agent.get_event_failed())
        raise
    except NodeCheckFailedError:
        is_node_check_failed = True
        raise
    except Exception:
        events.record(agent.get_event_failed())
        raise
    finally:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        client = MasterClient.singleton_instance()
        if (
            (exc_type is not None)
            or (result is not None and result.is_failed())
        ) and not is_node_check_failed:
            client.report_failed_exited()
            logger.info("Failed and exit.")
        elif is_node_check_failed:
            logger.info("Node check failed and exit.")

        if shutdown_rdzv:
            spec.rdzv_handler.shutdown()
        agent.stop_executor()
        monitor.stop()


def _create_worker_spec(
    node_rank: int,
    rdzv_name: str,
    config: ElasticLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
):
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        **config.rdzv_configs,
    )
    master_addr = _get_local_ip()
    rdzv_handler = MasterRendezvousHandler(
        rdzv_name,
        node_rank,
        rdzv_parameters,
        local_world_size=config.nproc_per_node,
    )
    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_handler,
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        master_addr=master_addr,
    )

    if version_less_than_230():
        spec.redirects = config.redirects
        spec.tee = config.tee
    return spec


class NodeCheckElasticAgent(ElasticTrainingAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers. This agent will run 2 rounds allgather
    to check network available.
    Round 0: the job master splits nodes into groups and each group contains
        two nodes. The node in each group will execute an allgather task and
        report its result to the master. For example, a job has 4 nodes and
        groups are [{0, 1}, {2, 3}]. Assuming that the allgather task in the
        1st group fails, the result is {0:False, 1:False, 2:True, 3:True}
        where the node 0, 1 are abnormal.
    Round 1: the master will group the abnormal node with a normal node like
        [{0, 2}, {1, 3}]. Then, the node executes an allgather task again.
        If the result is {0:True, 1:False, 2:False, 3:True}, the node-1
        breakdowns.
    """

    def __init__(
        self,
        node_rank,
        config,
        entrypoint,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
        check_round=1,
    ):
        super().__init__(
            node_rank,
            config,
            entrypoint,
            spec,
            start_method,
            exit_barrier_timeout,
            log_dir,
            with_diagnostician=False,
        )
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="node_check_")
        self._check_round = check_round
        self._config: ElasticLaunchConfig = config

    def _get_check_node_timeout(self):
        return JobConstant.MASTER_CLIENT_CHECK_NODE_TIMEOUT

    def run(self, role: str = DEFAULT_ROLE) -> bool:
        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            f"[{role}] starting node-check workers for entrypoint: "
            f"{spec.get_entrypoint_name()}"
        )
        success = False
        fault_nodes = []
        stragglers = []
        for i in range(self._check_round):
            result, elapsed_time = self._run_node_check(
                timeout=JobConstant.NODE_CHECK_TIMEOUT
            )
            elapsed_time = round(elapsed_time, 3)
            logger.info(
                f"Network check time of round {i} is {elapsed_time}"
                f" and succeed is {result}."
            )

            success = success or result
            status = (
                NodeEventType.NODE_CHECK_SUCCEEDED
                if success
                else NodeEventType.NODE_CHECK_FAILED
            )
            self._client.report_network_check_status(
                self._node_rank,
                status,
                elapsed_time,
            )

            fault_nodes, fault_reason = self._client.check_fault_node(
                timeout=self._get_check_node_timeout()
            )
            stragglers, straggler_reason = self._client.check_straggler(
                timeout=self._get_check_node_timeout()
            )
            logger.info(
                f"Fault nodes are: {fault_nodes} with {fault_reason} "
                f" and stragglers are: {stragglers} with {straggler_reason}"
            )
            self._stop_workers(self._worker_group)
            if fault_nodes or (stragglers and self._config.exclude_straggler):
                total_worker_num = len(self._client.get_running_nodes())
                if total_worker_num <= 3:
                    # If the number of nodes <= 3, we cannot determine which
                    # node if fault because there is no normal node in the job
                    # to execute allgather tasks with the two nodes.
                    logger.warning(
                        "No need for another round of network "
                        "check because the nodes is less than 3."
                    )
                    raise NodeCheckFailedError("This node is down.")
                else:
                    # Run the next round check to detect the fault node.
                    time.sleep(JobConstant.NODE_CHECK_NEXT_ROUND_TIMEOUT)
                    continue
            else:
                return success

        if self._node_rank in fault_nodes:
            self._client.report_failures(
                NodeErrorMessage.NETWORKER_ERROR,
                level=TrainingExceptionLevel.NODE_ERROR,
            )
            raise NodeCheckFailedError("This node is down.")
        elif self._node_rank in stragglers:
            logger.warning("This node is a straggler!")
            if self._config.exclude_straggler:
                raise NodeCheckFailedError(
                    "The node is a straggler and exits."
                )
        return success

    def _run_node_check(self, monitor_interval=3, timeout=300):
        self._initialize_workers(self._worker_group)
        start = time.time()
        succeed = False
        time_record_dir = ConfigPath.NETWORK_CHECK_DATA_DIR
        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state
            if state == WorkerState.HEALTHY:
                if time.time() - start > timeout:
                    logger.error(f"Timeout {timeout} to check network.")
                    break
                continue
            elif state == WorkerState.SUCCEEDED or self._check_finished(
                time_record_dir
            ):
                succeed = True
                break
            else:
                break

        if succeed:
            elapsed_time = self._get_node_check_time(time_record_dir)
        else:
            elapsed_time = 3600
        return succeed, elapsed_time

    def _check_finished(self, result_dir):
        if not os.path.exists(result_dir):
            return False
        num = len(os.listdir(result_dir))
        self._worker_group.workers
        return num == len(self._worker_group.workers)

    def _get_node_check_time(self, result_dir):
        elapsed_time = 0
        if not os.path.exists(result_dir):
            return elapsed_time
        for filename in os.listdir(result_dir):
            path = os.path.join(result_dir, filename)
            with open(path, "r") as f:
                data = f.read()
                if not data:
                    continue
                data = json.loads(data)
                elapsed_time = max(elapsed_time, data.get("time", 0))
        shutil.rmtree(result_dir, ignore_errors=True)
        return elapsed_time


def _create_check_agent(
    config: ElasticLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
    rdzv_name: str,
    check_round: int,
):
    """Create a agent to launch sub-processes."""
    config = copy.deepcopy(config)

    # Disable checking network when execute tasks to check network.
    config.network_check = False
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning(
            f"config has no run_id, generated a random run_id: {run_id}"
        )
        config.run_id = run_id

    entrypoint_name = _get_entrypoint_name(entrypoint, args)
    node_rank = env_utils.get_node_rank()

    logger.info(
        f"Starting node-check agent with launch configs:\n"
        f"  entrypoint       : {entrypoint_name}\n"
        f"  min_nodes        : {config.min_nodes}\n"
        f"  max_nodes        : {config.max_nodes}\n"
        f"  nproc_per_node   : {config.nproc_per_node}\n"
        f"  run_id           : {config.run_id}\n"
        f"  rdzv_backend     : {config.rdzv_backend}\n"
        f"  rdzv_configs     : {config.rdzv_configs}\n"
        f"  max_restarts     : {config.max_restarts}\n"
        f"  monitor_interval : {config.monitor_interval}\n"
        f"  log_dir          : {config.log_dir}\n"
        f"  metrics_cfg      : {config.metrics_cfg}\n"
    )

    spec = _create_worker_spec(
        node_rank=node_rank,
        rdzv_name=rdzv_name,
        config=config,
        entrypoint=entrypoint,
        args=args,
    )
    spec.max_restarts = 0  # Do not restart the check task.

    agent = NodeCheckElasticAgent(
        node_rank=node_rank,
        config=config,
        entrypoint=entrypoint,
        spec=spec,
        start_method=config.start_method,
        check_round=check_round,
    )
    return agent


def node_health_check(
    config: ElasticLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> bool:
    agent = _create_check_agent(
        config,
        entrypoint,
        args,
        RendezvousName.NETWORK_CHECK,
        check_round=2,
    )

    metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
    result = agent.run()
    logger.info("Network check result is %s", result)
    return result


def comm_perf_check(
    config: ElasticLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> bool:
    """Check the allreduce algorithm bandwidth and bus bandwidth."""
    agent = _create_check_agent(
        config,
        entrypoint,
        args,
        RendezvousName.ELASTIC_TRAINING,
        check_round=1,
    )

    metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
    result = agent.run()
    logger.info("Network check result is %s", result)
    return result


def run_network_check(config: ElasticLaunchConfig, entrypoint):
    if config.accelerator == Accelerators.NVIDIA_GPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.nvidia_gpu"]
    elif config.accelerator == Accelerators.ASCEND_NPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.ascend_npu"]
    else:
        logger.warning(f"Unsupported accelerator chip {config.accelerator}.")
        return True
    for _ in range(2):
        # If network fails because other abnormal node, We
        # will retry to check network after the new node is starting.
        # DLRover will replace the abnormal node with a new node.
        success = node_health_check(
            config=config, entrypoint=entrypoint, args=cmd_args
        )
        if success:
            break
        else:
            logger.error(
                "Network of the cluster is not available "
                "because of abnormal node."
            )
    if success and config.comm_perf_test:
        comm_perf_check(config=config, entrypoint=entrypoint, args=cmd_args)
    return success
