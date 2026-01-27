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
import copy
import functools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

from dlrover.python.elastic_agent.monitor.resource import (
    get_gpu_stats,
    get_hpu_stats,
)
import dlrover.python.util.store_util as store_util
from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    Accelerators,
    AscendConstants,
    ConfigPath,
    JobConstant,
    NodeEnv,
    NodeEventType,
    NodeExitDescription,
    RendezvousName,
    TrainingExceptionLevel,
    EventReportConstants,
    ScriptPath,
    NodeExitReason,
    RendezvousErrorType,
)
from dlrover.python.common.error import ProcessError
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
    NodeAction,
    JobAction,
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
from dlrover.python.elastic_agent.torch.dynamic_failover import (
    DynamicAgentFailoverExtension,
)
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore
from dlrover.python.training_event import DLRoverAgentEvent
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
    version_less_than_280,
)

_agent_evt = DLRoverAgentEvent().singleton_instance()


try:
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except (ModuleNotFoundError, ImportError):  # noqa: F841
    pass

try:
    from dlrover.python.common import musa_patch  # noqa: F401
except Exception:
    pass

__all__ = ["launch_agent"]
_DLROVER_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"


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
                "Can not resolve host IP. Use default '127.0.0.1' instead."
            )
            local_ip = "127.0.0.1"
    return local_ip


class RendezvousOutSyncError(Exception):
    pass


class JobStoppingError(Exception):
    pass


class NodeCheckFailedError(RuntimeError):
    pass


class RendezvousTimeoutError(RuntimeError):
    pass


class StopWorkerTimeoutError(RuntimeError):
    pass


class LogConfig:
    _log_dir: Optional[str] = None
    _redirects: Union[Std, Dict[int, Std]] = Std.NONE
    _tee: Union[Std, Dict[int, Std]] = Std.NONE

    @classmethod
    def _parse_std_value(cls, val) -> Std:
        if val is None:
            return Std.NONE

        if isinstance(val, str):
            try:
                return Std.from_str(val)
            except ValueError:
                return Std.NONE
        elif isinstance(val, Std):
            return val

        return Std.NONE

    def setup(self, log_dir, redirects=None, tee=None):
        if not log_dir:
            return

        self._log_dir = log_dir

        redirects = self._parse_std_value(redirects)
        tee = self._parse_std_value(tee)

        if redirects == Std.NONE and tee == Std.NONE:
            # override default when log dir is specified
            self._redirects = Std.ALL
            self._tee = Std.ALL
        else:
            self._redirects = redirects
            self._tee = tee

        logger.debug(
            f"Log config: {self._log_dir}-{self._redirects}-{self._tee}"
        )

    @property
    def log_dir(self) -> Optional[str]:
        if not self._log_dir:
            tmp_dir = "/tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            return tmp_dir
        return self._log_dir

    @property
    def redirects(self) -> Union[Std, Dict[int, Std]]:
        return self._redirects

    @property
    def tee(self) -> Union[Std, Dict[int, Std]]:
        return self._tee

    @property
    def logs_specs(self):
        if version_less_than_230():
            return {
                "log_dir": self.log_dir,
                "redirects": self.redirects,
                "tee": self.tee,
            }
        else:
            from torch.distributed.elastic.multiprocessing import (
                DefaultLogsSpecs,
            )

            log_specs = DefaultLogsSpecs(
                log_dir=self.log_dir, redirects=self.redirects, tee=self.tee
            )
            log_specs._run_log_dir = self.log_dir
            return log_specs


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
        numa_affinity: whether numa affinity is enabled.
        membind_policy: membind policy for numa affinity.
        ucp_device_type: device type for unified checkpoint.
        dynamic_failover_extension: extend implementation for dynamic failover.
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
    log_config: LogConfig = LogConfig()
    training_log_file: str = ""
    failure_node_errors: str = ""
    numa_affinity: bool = False
    membind_policy: str = "none"
    ucp_device_type: str = "cpu"
    dynamic_failover_extension: Optional[DynamicAgentFailoverExtension] = None

    def get_log_dir(self):
        return self.log_config.log_dir

    def get_log_tee(self):
        return self.log_config.tee

    def get_log_redirects(self):
        return self.log_config.redirects

    def get_log_specs(self):
        return self.log_config.logs_specs

    def setup_log(self, log_dir, redirects=None, tee=None):
        if log_dir:
            logger.info(f"Initiate specified log directory: {log_dir}.")
            self.log_config.setup(log_dir, redirects=redirects, tee=tee)
        else:
            logger.info("No specified log directory is configured.")

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
        if "mthreads" in device:
            self.accelerator = Accelerators.MTHREADS_GPU
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

    def to_json(self):
        return {
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "nproc_per_node": self.nproc_per_node,
            "rdzv_backend": self.rdzv_backend,
            "rdzv_endpoint": self.rdzv_endpoint,
            "rdzv_configs": json.dumps(self.rdzv_configs),
            "max_restarts": self.max_restarts,
            "accelerator": self.accelerator,
        }


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
        self._store = MasterKVStore(self._name, timedelta(seconds=300))
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

    def _get_rdzv_error_data(self, err_type="", err_msg="", elapsed_time=-1):
        return json.dumps(
            {
                "rdzv_name": self._name,
                "node_rank": self._node_rank,
                "err_type": err_type,
                "err_message": err_msg,
                "elapsed_time": elapsed_time,
            }
        )

    def next_rendezvous(self):
        """The handler will periodically query the world from the master until
        the world is not empty. The world is a dictionary like
        {0: 8, 1: 8, 2: 8} where the key is the node ID and the value is
        the local world size. The handler can get its rank by the position
        of it node ID in the world.
        """
        start_join = time.time()
        node_name = os.getenv(NodeEnv.POD_NAME, "")
        msg = (
            f"The node '{node_name}' with rank {self._node_rank} attempts to "
            f"join the next round of the rendezvous {self._name} "
            f"with timeout {self.join_timeout}."
        )
        logger.info(msg)
        _rdzv_evt = _agent_evt.rendezvous(
            rendezvous_type=self._name,
            node_name=node_name,
            node_rank=self._node_rank,
            timeout=self.join_timeout,
        )
        _rdzv_evt.begin()

        self._join_rendezvous()

        start_pending = 0
        while True:
            self._check_network_rdzv()
            round, group, world = self._client.get_comm_world(
                self._name, self._node_rank
            )
            if world:
                if self._node_rank in world:
                    break
                else:
                    if start_pending == 0:
                        logger.info(
                            "The node is not in the world and waits for more nodes."
                        )
                        start_pending = time.time()
                    time.sleep(JobConstant.RENDEZVOUS_DEFAULT_INTERVAL)
                    start_join = time.time()
                    if start_join - start_pending > self.pend_timeout:
                        err_msg = (
                            f"Timeout {self.pend_timeout}s to wait more nodes"
                        )
                        self._report_failure(
                            self._get_rdzv_error_data(
                                RendezvousErrorType.PEND_TIMEOUT,
                                err_msg,
                                int(self.pend_timeout),
                            ),
                            level=TrainingExceptionLevel.RDZV_ERROR,
                            rank0_only=False,
                        )
                        _rdzv_evt.fail(error=err_msg)
                        raise RendezvousTimeoutError(err_msg)
                    continue
            elif time.time() - start_join > self.join_timeout:
                timeout = self.join_timeout
                err_msg = (
                    f"Timeout {timeout}s to wait the enough nodes "
                    "to complete rendezvous."
                )
                self._report_failure(
                    self._get_rdzv_error_data(
                        RendezvousErrorType.JOIN_TIMEOUT,
                        err_msg,
                        self.join_timeout,
                    ),
                    level=TrainingExceptionLevel.RDZV_ERROR,
                    rank0_only=False,
                )
                _rdzv_evt.fail(error=err_msg)
                raise RendezvousTimeoutError(err_msg)
            time.sleep(JobConstant.RENDEZVOUS_DEFAULT_INTERVAL)
        rank = list(world.keys()).index(self._node_rank)
        world_size = len(world)
        logger.info(
            f"The node {node_name} has joined round {round} of "
            f"the {self._name} rendezvous as rank {rank} in a world of size "
            f"{world_size}."
        )
        if (
            self._name == RendezvousName.TRAINING
            and world_size < self._rdzv_params.max_nodes
        ):
            err_msg = f"Scale down the number of nodes to {world_size}"
            self._report_failure(
                self._get_rdzv_error_data("", err_msg),
                level=TrainingExceptionLevel.WARNING,
            )

        _rdzv_evt.success(
            round=round,
            rank=rank,
            world_size=world_size,
        )
        store = self._get_store(round, group)
        return store, world

    def _check_network_rdzv(self):
        """
        1. The worker need to exit the elastic-training rendezvous if there are
        workers to join the network-check rendezvous.
        2. If job is stopping, raise exception and stop rendezvous
        """
        num = self._client.num_nodes_waiting(RendezvousName.NETWORK_CHECK)

        if self._name == RendezvousName.TRAINING:
            if num > 0:
                raise RendezvousOutSyncError(
                    "Some workers join the network-check rendezvous"
                    "not the elastic-training rendezvous."
                )

        if num < 0:
            raise JobStoppingError("Exit rendezvous when job is stopping")

    def _report_failure(self, err_msg, level, rank0_only=True):
        if rank0_only and not self._node_rank == 0:
            return
        else:
            self._client.report_failures(err_msg, 0, level)

    def _get_store(self, round, group) -> Store:
        key_prefix = f"torch.rendezvous.{self._name}.{round}.{group}"
        return PrefixStore(key_prefix, self._store)

    def num_nodes_waiting(self):
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
        return True


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

    node_device_check = False

    def __init__(
        self,
        node_rank,
        config: ElasticLaunchConfig,
        entrypoint,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        training_log_file: str = "",
        failure_node_errors: str = "",
        with_diagnostician: bool = True,
    ):
        if version_less_than_230():
            super().__init__(
                spec=spec,
                exit_barrier_timeout=exit_barrier_timeout,
            )
            # compatible
            # https://github.com/pytorch/pytorch/blob/39901f229520a5256505ec24782f716ee7ddc843/torch/distributed/elastic/agent/server/local_elastic_agent.py#L148C9-L148C22
            self._log_dir = config.get_log_dir()
        else:
            logger.info(
                "Setup logging configuration for torch version>=230 with "
                f"log_dir: {config.get_log_dir()}, "
                f"redirections: {config.get_log_redirects()}, "
                f"tee: {config.get_log_tee()}, log_specs: {config.get_log_specs().__dict__}"
            )
            super().__init__(
                spec=spec,
                logs_specs=config.get_log_specs(),
                exit_barrier_timeout=exit_barrier_timeout,
            )
        self._node_rank = node_rank
        self._config = config
        self._entrypoint = entrypoint
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
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
                training_log_file=training_log_file,
                errors=failure_node_errors,
                node_rank=node_rank,
                local_world_size=config.nproc_per_node,
                dynamic_failover_extension=config.dynamic_failover_extension,
            )
        else:
            self._diagnose_agent = None

        self._agent_context = get_agent_context()
        self._rank_cpu_affinity = {}
        if self._config.numa_affinity:
            for rank in range(self._config.nproc_per_node):
                if self._config.accelerator == Accelerators.ASCEND_NPU:
                    self._rank_cpu_affinity[rank] = get_npu_affinity(rank)
                else:
                    self._rank_cpu_affinity[rank] = get_gpu_affinity(rank)
                logger.info(
                    f"get rank {rank} affinity: {self._rank_cpu_affinity[rank]}"
                )

    @classmethod
    def is_device_checked(cls):
        return cls.node_device_check

    @classmethod
    def set_device_checked(cls):
        cls.node_device_check = True

    @classmethod
    def reset_device_checked(cls):
        cls.node_device_check = False

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

    def _graceful_exit_workers(self, worker_group: WorkerGroup):
        logger.info("try to gracefully exit workers")
        if self._pcontext is None:
            logger.warning(
                "_pcontext is None, cannot send graceful exit signal"
            )
            return
        pid_set = set(self._pcontext.pids().values())
        pc_pids = list(pid_set)
        if not pc_pids:
            logger.warning(
                "No worker PIDs available, cannot send graceful exit signal"
            )
            return
        pc_pid = pc_pids[0]
        try:
            os.kill(pc_pid, signal.SIGUSR1)
        except Exception as e:
            logger.warning(f"error when kill {pc_pid}: {str(e)}")

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
        """
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

        if self._diagnose_agent is not None:
            logger.info(
                f"[{spec.role}] Reset event collector after rendezvous"
            )
            self._diagnose_agent.reset_atorch_collector()

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

    def _get_ranks(
        self,
        role_infos: List[_RoleInstanceInfo],
        role_idx: int,
        start_idx: int = 0,
        end_idx: int = -1,
    ) -> Tuple[int, List[int]]:
        if end_idx == -1:
            end_idx = len(role_infos)
        prefix_sum = 0
        total_sum = 0
        for idx in range(start_idx, end_idx):
            if role_idx > idx:
                prefix_sum += role_infos[idx].local_world_size
            total_sum += role_infos[idx].local_world_size
        return (
            total_sum,
            list(
                range(
                    prefix_sum,
                    prefix_sum + role_infos[role_idx].local_world_size,
                )
            ),
        )

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
        return workers

    def _initialize_workers(self, worker_group, max_errors=3):
        logger.info(
            f"Start initializing training({self.__class__.__name__}) workers."
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
                    raise RendezvousTimeoutError(
                        "Timeout to wait for new nodes."
                    )
            except (
                NodeCheckFailedError,
                RendezvousTimeoutError,
                StopWorkerTimeoutError,
                JobStoppingError,
            ) as e:
                raise e
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
                logger.info(
                    f"Finish initializing training({self.__class__.__name__}) workers."
                )
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
                elif version_less_than_280():
                    super()._stop_workers(worker_group, is_restart)
                else:
                    super()._stop_workers(worker_group)
                self._stop_orphan_workers(worker_group)

            signal.alarm(0)
        except StopWorkerTimeoutError as te:
            logger.error(str(te))
            raise te
        finally:
            signal.alarm(0)

    def _stop_timeout_handler(self, signum, frame):
        current_pid = os.getpid()
        current_pgid = os.getpgid(current_pid)

        def get_processes_by_pgid(pgid):
            target_processes = []
            for proc in psutil.process_iter(
                ["pid", "ppid", "name", "cmdline"]
            ):
                try:
                    if os.getpgid(proc.pid) == pgid:
                        cmdline = (
                            " ".join(proc.cmdline())
                            if proc.cmdline()
                            else proc.name()
                        )
                        target_processes.append(
                            {
                                "pid": proc.pid,
                                "ppid": proc.ppid(),
                                "name": proc.name(),
                                "cmdline": cmdline,
                                "kernel_stack": env_utils.get_kernel_stack(
                                    proc.pid
                                )[1],
                                "user_stack": env_utils.get_user_stack_pyspy(
                                    proc.pid
                                )[1],
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return target_processes

        try:
            # get all the subprocess's pgid by current pid
            child_pids = env_utils.get_all_child_pids(current_pid)
            child_pgids = set(
                [os.getpgid(child_pid) for child_pid in child_pids]
            )

            # kill child pg 1st
            if child_pids and child_pgids:
                for child_pgid in child_pgids:
                    target_processes = get_processes_by_pgid(child_pgid)

                    # print target processes' stack info and do killing
                    logger.warning(
                        "Use pkill to kill all sub-processes in 'stop_timeout_handler', "
                        f"Target pgid: {child_pgid}, processes: {target_processes}"
                    )
                    subprocess.run(
                        ["pkill", "-9", "-g", str(child_pgid)],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

            time.sleep(1)

            # if still remain child process, kill current pgid
            child_pids = env_utils.get_all_child_pids(current_pid)
            if child_pids:
                logger.warning(
                    f"Still remaining process after pkill child process in 'stop_timeout_handler': {child_pids}"
                )
                target_processes = get_processes_by_pgid(current_pgid)
                logger.warning(
                    "Use pkill to kill all processes(including current process) in 'stop_timeout_handler', "
                    f"Target pgid: {current_pgid}, processes: {target_processes}"
                )
                subprocess.run(
                    ["pkill", "-9", "-g", str(current_pgid)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
        except Exception:
            logger.exception(
                "Unexpected error in stop_timeout_handler when killing process."
            )

        raise StopWorkerTimeoutError(
            "Timed out waiting for stopping workers, forcefully kill all sub-processes."
        )

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

        if self._config.numa_affinity and isinstance(spec.entrypoint, str):
            os.environ["DLROVER_MEMBIND_POLICY"] = self._config.membind_policy
            logger.info(
                f"WorkerGroup before numa affinity: {self._worker_group.spec}"
            )
            self._worker_group.spec.args = (
                self._worker_group.spec.entrypoint,
            ) + self._worker_group.spec.args
            self._worker_group.spec.entrypoint = ScriptPath.RUN_AFFINITY_SCRIPT
            logger.info(
                f"WorkerGroup after numa affinity: {self._worker_group.spec}"
            )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

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
                    if version_less_than_240():
                        self._dlrover_exit_barrier()
                    else:
                        self._exit_barrier()
                    logger.info("Barrier exited.")

                    self._wait_async_saver()
                    logger.info("Async saver stopped.")
                except Exception as e:
                    logger.warning(f"Unexpected exception when ending: {e}")
                finally:
                    self._client.report_succeeded_exited()
                    logger.info("Succeeded and exit.")
                    _agent_evt.process_succeeded(
                        node_rank=self._node_rank,
                        return_values=f"{run_result.return_values}",
                        state=f"{run_result.state.name}",
                    )

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
                            node_id=env_utils.get_node_id(),
                            node_type=env_utils.get_node_type(),
                            instance=DiagnosisConstant.LOCAL_INSTANCE,
                            action_type=DiagnosisActionType.RESTART_WORKER,
                        )
                    else:
                        action = NodeAction(
                            node_id=env_utils.get_node_id(),
                            node_type=env_utils.get_node_type(),
                            instance=DiagnosisConstant.LOCAL_INSTANCE,
                            action_type=DiagnosisActionType.RELAUNCH_WORKER,
                        )

                if action.action_type == DiagnosisActionType.RESTART_WORKER:
                    _agent_evt.process_restart(
                        node_rank=self._node_rank,
                        restart_count=self._restart_count,
                        remaining_restarts=self._remaining_failovers,
                        state=f"{run_result.state.name}",
                        return_values=f"{run_result.return_values}",
                        failures=f"{run_result.failures}",
                    )
                elif action.action_type == DiagnosisActionType.RELAUNCH_WORKER:
                    _agent_evt.process_fail(
                        node_rank=self._node_rank,
                        restart_count=self._restart_count,
                        remaining_restarts=self._remaining_failovers,
                        state=f"{run_result.state.name}",
                        return_values=f"{run_result.return_values}",
                        failures=f"{run_result.failures}",
                    )
                elif action.action_type == DiagnosisActionType.JOB_ABORT:
                    _agent_evt.job_abortion(
                        node_rank=self._node_rank,
                        reason=action.reason,
                        state=f"{run_result.state.name}",
                        return_values=f"{run_result.return_values}",
                        failures=f"{run_result.failures}",
                    )
                elif action.action_type == DiagnosisActionType.JOB_RESTART:
                    _agent_evt.job_restart(
                        node_rank=self._node_rank,
                        reason=action.reason,
                        state=f"{run_result.state.name}",
                        return_values=f"{run_result.return_values}",
                        failures=f"{run_result.failures}",
                    )

                self._process_diagnosis_action(action)

                if self._worker_group.state == WorkerState.FAILED:
                    return run_result

            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                if self._membership_changed(role, rdzv_handler):
                    self._graceful_exit_workers(
                        worker_group=self._worker_group
                    )
                    elastic_mode = os.getenv(
                        "DLROVER_TRAINING_ELASTIC_MODE", "base"
                    ).lower()

                    if (
                        self._worker_group.group_rank == 0
                        and elastic_mode == "ucp"
                    ):
                        self.set_rdzv_blocked(True)
                    logger.info(
                        f"self._worker_group.group_rank : {self._worker_group.group_rank}"
                    )
                    _agent_evt.process_restart_membership(
                        node_rank=self._node_rank,
                        restart_count=self._restart_count,
                        remaining_restarts=self._remaining_failovers,
                        state=f"{run_result.state.name}",
                        return_values=f"{run_result.return_values}",
                        failures=f"{run_result.failures}",
                    )
                    self._save_ckpt_to_storage()

                    if (
                        self._worker_group.group_rank == 0
                        and elastic_mode == "ucp"
                    ):
                        self.ucp()
                    if (
                        self._worker_group.group_rank == 0
                        and elastic_mode == "ucp"
                    ):
                        self.set_rdzv_blocked(False)
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] worker group in {state.name} state")

    def _process_diagnosis_action(self, action: DiagnosisAction):
        if not action:
            return
        action_type = action.action_type

        if isinstance(action, NodeAction):
            action.__class__ = NodeAction
            if action_type == DiagnosisActionType.RESTART_WORKER:
                logger.info(
                    f"Process diagnosis action: {action_type} {action.instance}"
                )
                if action.instance == DiagnosisConstant.LOCAL_INSTANCE:
                    self._remaining_failovers -= 1
                    logger.info(
                        f"Decrement remaining FO to {self._remaining_failovers}"
                    )
                self._restart_workers(self._worker_group)
            elif action_type == DiagnosisActionType.RELAUNCH_WORKER:
                logger.info(
                    f"Process diagnosis action: {action_type} {action.instance}"
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
        elif isinstance(action, JobAction):
            logger.info(
                f"Report job action to master for following process: {action_type}."
            )
            self._client.report_action(action)

    def _check_and_process_diagnosis_action(self):
        for instance in [
            DiagnosisConstant.LOCAL_INSTANCE,
            DiagnosisConstant.ANY_INSTANCE,
        ]:
            action = self._agent_context.next_diagnosis_action(instance)
            if isinstance(action, NoAction):
                continue
            logger.info(f"Start processing diagnosis action: {action}")
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

    def ucp(self):
        """
        Do universal checkpoint.

        Semantics:
        - UCP must always operate on a successfully saved checkpoint.
        - start_saving_step is only used to detect and wait for a new save.
        """
        try:
            saver: AsyncCheckpointSaver = AsyncCheckpointSaver.get_ckpt_saver()
            if not saver:
                return

            start_time = time.time()
            max_wait_time = int(os.getenv("DLROVER_UCP_MAX_WAIT_TIME", "60"))
            wait_time = int(os.getenv("DLROVER_UCP_WAIT_TIME", "30"))

            # Initial snapshot
            checkpoint_dir, last_success_step = (
                saver.get_latest_success_save_dir()
            )
            start_saving_step = saver.get_latest_start_saving_step()

            # If nothing has ever been saved or started, nothing to UCP
            if last_success_step is None and start_saving_step is None:
                logger.info("No checkpoint has ever been started, skip ucp.")
                return
            target_step = None
            need_new_saving = False

            while True:
                checkpoint_dir, success_step = (
                    saver.get_latest_success_save_dir()
                )
                start_saving_step = saver.get_latest_start_saving_step()
                elapsed_time = time.time() - start_time

                # Detect that a NEW checkpoint saving has actually started
                if (
                    start_saving_step is not None
                    and last_success_step is not None
                    and start_saving_step > last_success_step
                ):
                    target_step = start_saving_step
                    need_new_saving = True

                # If we have seen the new saving and it completed successfully
                if need_new_saving and success_step == target_step:
                    break

                # No new checkpoint triggered within wait_time
                if not need_new_saving and elapsed_time > wait_time:
                    logger.info(
                        "No new checkpoint triggered, "
                        "using last successful step %s for ucp.",
                        success_step,
                    )
                    break

                # Timeout waiting for the new checkpoint
                if elapsed_time > max_wait_time:
                    logger.warning(
                        "Timeout waiting for checkpoint step %s. "
                        "Fallback to last successful step %s.",
                        target_step,
                        success_step,
                    )
                    break

                time.sleep(1)

            # -------- Final decision: ONLY use successful checkpoint --------
            checkpoint_dir, step = saver.get_latest_success_save_dir()
            if checkpoint_dir is None or step is None:
                logger.error("No successful checkpoint available for ucp.")
                return

            input_dir = os.path.join(
                checkpoint_dir,
                f"checkpoint-{step}",
                f"global_step{step}",
            )
            output_dir = os.path.join(
                checkpoint_dir,
                f"checkpoint-{step}",
                "ucp",
            )

            res = saver.ucp(
                input_dir,
                output_dir,
                self._config.ucp_device_type,
            )
            if res:
                with open(
                    os.path.join(checkpoint_dir, "ucp.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(str(step))

        except Exception:
            logger.exception("ucp failed")
            raise

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

    def set_rdzv_blocked(self, blocked, reason=""):
        return self._client.set_rdzv_blocked(blocked=blocked, reason=reason)

    def _dlrover_exit_barrier(self):
        logger.info(
            f"Local worker group finished {self._worker_group.state}. "
            f"Waiting {self._exit_barrier_timeout} seconds "
            f"for other agents to finish"
        )
        start = time.time()
        try:
            store_util.barrier(
                self._store,
                self._worker_group.group_world_size,
                key_prefix=_DLROVER_TERMINAL_STATE_SYNC_ID,
                barrier_timeout=self._exit_barrier_timeout,
            )
            logger.info(
                f"Done waiting for other agents. Elapsed: {time.time() - start} seconds"
            )
        except SignalException as e:
            logger.warning(f"Got termination signal: {e.sigval}")
            raise
        except Exception:
            logger.error(
                f"Error waiting on exit barrier. Elapsed: {time.time() - start} seconds"
            )


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
        f"Launching agent entrypoint: {entrypoint}, args: {args}, "
        f"name: {entrypoint_name}, rank: {node_rank}"
    )

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
        f"  log_dir          : {config.get_log_dir()}\n"
        f"  metrics_cfg      : {config.metrics_cfg}\n"
        f"  training_log     : {config.training_log_file}\n"
        f"  failure_errors   : {config.failure_node_errors}\n"
        f"  numa_affinity    : {config.numa_affinity}\n"
        f"  accelerator      : {config.accelerator}\n"
        f" ucp_device_type   : {config.ucp_device_type}\n"
    )

    _agent_evt.start(args=vars(config))

    _set_paral_config()
    monitor = TorchTrainingMonitor(
        ConfigPath.RUNTIME_METRICS, config.accelerator
    )
    monitor.start()

    spec = _create_worker_spec(
        node_rank=node_rank,
        rdzv_name=RendezvousName.TRAINING,
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
        training_log_file=config.training_log_file,
        failure_node_errors=config.failure_node_errors,
        exit_barrier_timeout=900,
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

        _agent_evt.exit(success=True)
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

        if result is None:
            _agent_evt.exit(
                success=False,
                exc_type=f"{exc_type}",
                exc_value=f"{exc_value}",
                exc_traceback=f"{exc_traceback}",
            )
        else:
            _agent_evt.exit(
                success=False,
                exc_type=f"{exc_type}",
                exc_value=f"{exc_value}",
                exc_traceback=f"{exc_traceback}",
                state=f"{result.state.name}",
                return_values=f"{result.return_values}",
                failures=f"{result.failures}",
            )

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

    # for torch < 230, the tee and redirects config for log is located in spec
    if version_less_than_230():
        spec.redirects = config.get_log_redirects()
        spec.tee = config.get_log_tee()
        logger.info(
            "Setup logging configuration for torch version<230 with "
            f"log_dir: {config.get_log_dir()}, "
            f"redirections: {config.get_log_redirects()}, "
            f"tee: {config.get_log_tee()}"
        )
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
        check_round=1,
    ):
        super().__init__(
            node_rank,
            config,
            entrypoint,
            spec,
            start_method,
            exit_barrier_timeout,
            with_diagnostician=False,
        )
        self._check_round = check_round
        self._config: ElasticLaunchConfig = config

    def network_check_evt(self, round, node_rank):
        return _agent_evt.network_check(
            round=round,
            node_rank=node_rank,
        )

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
            evt = self.network_check_evt(i, self._node_rank)
            evt.begin()

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
            if success:
                evt.success(
                    result=f"{result}",
                    status=f"{status}",
                    elapsed_time=f"{elapsed_time}",
                )
            else:
                evt.fail(
                    result=f"{result}",
                    status=f"{status}",
                    elapsed_time=f"{elapsed_time}",
                )

            fault_nodes, fault_reason = self._client.check_fault_node(
                timeout=self._get_check_node_timeout()
            )
            stragglers, straggler_reason = self._client.check_straggler(
                timeout=self._get_check_node_timeout()
            )
            logger.info(
                f"Fault nodes are: {fault_nodes} with reason: {fault_reason}, "
                f"and stragglers are: {stragglers} with reason: {straggler_reason}"
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
                    raise NodeCheckFailedError(
                        NodeExitDescription.NODE_FAILED_MSG
                    )
                else:
                    # Run the next round check to detect the fault node.
                    time.sleep(JobConstant.NODE_CHECK_NEXT_ROUND_TIMEOUT)
                    continue
            else:
                return success

        if self._node_rank in fault_nodes:
            self._client.report_failures(
                NodeExitReason.CHECK_FAIL,
                level=TrainingExceptionLevel.NODE_ERROR,
            )
            raise NodeCheckFailedError(NodeExitDescription.NODE_FAILED_MSG)
        elif self._node_rank in stragglers:
            logger.warning("This node is a straggler!")
            if self._config.exclude_straggler:
                raise NodeCheckFailedError(
                    NodeExitDescription.NODE_STRAGGLE_MSG
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
        f"  log_dir          : {config.get_log_dir()}\n"
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
        RendezvousName.TRAINING,
        check_round=1,
    )

    metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
    result = agent.run()
    logger.info("Network check result is %s", result)
    return result


def _check_device(config: ElasticLaunchConfig):
    if ElasticTrainingAgent.is_device_checked():
        logger.info("Skip device check for not 1st time starting.")
        return

    check_result = True, -1
    ElasticTrainingAgent.set_device_checked()

    if config.accelerator == Accelerators.NVIDIA_GPU:
        # for gpu or cpu
        device_stats = get_gpu_stats()
    elif config.accelerator == Accelerators.ASCEND_NPU:
        # for ascend
        device_stats = get_hpu_stats()
    else:
        logger.debug(
            f"Device type {config.accelerator} is not supported for checking."
        )
        return

    logger.debug(f"Device stats: {device_stats}")
    if not device_stats:
        logger.info("Skip device check for stats is empty.")
        return

    for device_stat in device_stats:
        index = device_stat.index

        # to avoid memory leak
        if device_stat.used_memory_mb / device_stat.total_memory_mb >= 0.3:
            logger.error(
                f"Device[{config.accelerator}] check[{index}] "
                "failed: occupied memory detected."
            )
            check_result = False, index
            break

    if not check_result[0]:
        client = MasterClient.singleton_instance()
        client.report_event(
            event_type=EventReportConstants.TYPE_WARN,
            instance=env_utils.get_hostname_and_ip()[0],
            action=EventReportConstants.ACTION_DEVICE_WARNING,
            msg="Device check failed",
            labels={"device": check_result[1]},
        )
        raise NodeCheckFailedError(NodeExitDescription.GPU_INVALID_MSG)

    logger.info(f"Device[{config.accelerator}] check succeeded.")


def run_network_check(config: ElasticLaunchConfig, entrypoint):
    _check_device(config)

    # matmul + comm check
    if config.accelerator == Accelerators.NVIDIA_GPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.nvidia_gpu"]
    elif config.accelerator == Accelerators.ASCEND_NPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.ascend_npu"]
    elif config.accelerator == Accelerators.MTHREADS_GPU:
        cmd_args = ["-m", "dlrover.trainer.torch.node_check.mthreads_gpu"]
    else:
        logger.warning(f"Unsupported accelerator chip {config.accelerator}.")
        return True
    for _round in range(2):
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
                "Network of the cluster is not available because of abnormal node."
            )
    if success and config.comm_perf_test:
        comm_perf_check(config=config, entrypoint=entrypoint, args=cmd_args)
    return success
