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
import signal
import socket
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

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
from torch.distributed.elastic.multiprocessing import PContext, SignalException
from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
)
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.api import RendezvousHandler
from torch.distributed.launcher.api import LaunchConfig, _get_entrypoint_name

from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    ConfigPath,
    NodeErrorMessage,
    NodeStatus,
    RendezvousName,
    TrainingMsgLevel,
)
from dlrover.python.common.grpc import (
    find_free_port_in_range,
    find_free_port_in_set,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.config.paral_config_tuner import (
    ParalConfigTuner,
)
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.elastic_agent.monitor.training import TorchTrainingMonitor
from dlrover.python.elastic_agent.torch.ckpt_saver import AsyncCheckpointSaver
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore

try:
    import torch_npu  # noqa: F401
except (ModuleNotFoundError, ImportError) as e:  # noqa: F841
    torch_npu = None

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
        local_ip = socket.gethostbyname(_get_fq_hostname())
    return local_ip


class RendezvousOutSyncError(Exception):
    pass


@dataclass
class ElasticLaunchConfig(LaunchConfig):
    """
    Creates a rendezvous config of elastic training.

    Args:
        network_check: whether to check the network avaliable before training.
        node_unit: the number of unit of nodes. The number of nodes must be
            a multiple of node_unit.
        auto_tunning: whether to auto-tune the parallelism configuration.
        exclude_straggler: The node will exit if it is a straggler in network
            check and exclude_straggler is True.
    """

    network_check: bool = False
    node_unit: int = 1
    auto_tunning: bool = False
    exclude_straggler: bool = False

    def set_node_unit(self, node_unit):
        """Set the number unint of ndoes."""
        self.node_unit = node_unit
        self.rdzv_configs["node_unit"] = node_unit


@dataclass
class ProcessError:
    local_rank: int
    exitcode: int
    message: str
    datetime: Any


class MasterRendezvousHandler(RendezvousHandler):
    """The rendzevous handler completes rendezvous by connecting
    with the ElasticJob master. The master will collect all nodes
    after the handler of all node agents calls `_join_rendezvous`.
    Then, the handler will get the communcation world from the master
    and assign ranks to the training process.

    Args:
        name: the name of rendezvous.
        node_rank: the node rank.
        rdzv_params: RendezvousParameters instance. We can set timeout of
            rendezvous in the rdzv_params.config. Now we set:
            join_timeout: the timeout to join the rendevous. The timeout
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
        self.join_timeout = int(rdzv_params.get("join_timeout", 600))
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
        """The handler will peroidically query the world from the master until
        the world is not empty. The world is a dictionary like
        like {0: 8, 1: 8, 2: 8} where the key is the node ID and the value is
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
                    logger.info(
                        "The node is not in the world "
                        "and waits for more nodes."
                    )
                    if start_pending == 0:
                        start_pending = time.time()
                    time.sleep(5)
                    start_join = time.time()
                    if start_join - start_pending > self.pend_timeout:
                        raise TimeoutError(
                            f"Timeout {self.pend_timeout}s to wait more nodes"
                        )
                    continue
            elif time.time() - start_join > self.join_timeout:
                timeout = self.join_timeout
                err_msg = f"Timeout {timeout}s to complete rendezvous."
                self._report_failure(
                    err_msg, level=TrainingMsgLevel.RDZV_ERROR
                )
                raise TimeoutError(err_msg)
            time.sleep(3)
        world = dict(sorted(world.items()))
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
            self._report_failure(err_msg, level=TrainingMsgLevel.WARNING)
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
    ):
        super().__init__(spec, exit_barrier_timeout)
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
            self._paral_config_tuner = ParalConfigTuner()
            self._paral_config_tuner.start()

        self._save_ckpt_executor = ThreadPoolExecutor(max_workers=1)
        self._save_ckpt_future = None

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

        master_addr, master_port = self._get_master_addr_port(store)

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
                logger.warn(e)
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
        my_role_info = role_infos[group_rank]
        worker_world_size, worker_global_ranks = self._get_ranks(
            role_infos, group_rank
        )
        role_infos = sorted(
            role_infos, key=functools.cmp_to_key(_RoleInstanceInfo.compare)
        )
        role_start_idx, role_end_idx = _RoleInstanceInfo.find_role_boundaries(
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

    def _initialize_workers(self, worker_group):
        while True:
            try:
                if self._config.network_check:
                    run_network_check(self._config, self._entrypoint)
                super()._initialize_workers(worker_group)
                # We need to register handler after starting workers because
                # the PContext start_worker will overwrite the handler.
                AsyncCheckpointSaver.register_signal_handler()
            except RendezvousOutSyncError:
                logger.info(
                    "Exit elastic-training rendezvous when there are "
                    "agents to join the network-check rendezvous."
                )
            else:
                break

    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        if torch_npu:
            logger.info("stop workers via SIGKILL")
            self._shutdown(death_sig=signal.SIGKILL)
        else:
            super()._stop_workers(worker_group)

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # Start a thread to save the checkpointing state dict from
        # the shared memory to the storage.
        AsyncCheckpointSaver.start_async_saving_ckpt()

        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            f"[{role}] starting workers for entrypoint: "
            f"{spec.get_entrypoint_name()}"
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
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
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                logger.error(f"The worker fails with {run_result.failures}")
                self._report_failure_to_master(run_result.failures)
                self._save_ckpt_to_storage()
                if self._remaining_failovers > 0:
                    logger.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_failovers}/{spec.max_restarts}"
                        f" attempts left; will restart worker group"
                    )
                    self._remaining_failovers -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                if self._membership_changed(role, rdzv_handler):
                    self._save_ckpt_to_storage()
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    def _save_ckpt_to_storage(self):
        """
        The agent can save the checkpointing state dict in the shared
        memory into the storage before restarting training processes.
        """
        saver: AsyncCheckpointSaver = AsyncCheckpointSaver.get_ckpt_saver()
        if saver:
            self._save_ckpt_future = self._save_ckpt_executor.submit(
                saver.save_shm_to_storage
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
            error_data, self._restart_count, TrainingMsgLevel.PROCESS_ERROR
        )

    def _restart_workers(self, worker_group: WorkerGroup):
        self._restart_count += 1
        self._remaining_restarts -= 1
        # Relase the shared memory lock before starting workers.
        AsyncCheckpointSaver.release_shm_lock()
        super()._restart_workers(worker_group)

    def _start_workers(self, worker_group: WorkerGroup):
        if self._save_ckpt_future:
            # Waiting the thread to save checkpoint finishes.
            self._save_ckpt_future.result(timeout=600)
        return super()._start_workers(worker_group)

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
        self._save_ckpt_executor.shutdown()


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
        f"Starting elastic_operator with launch configs:\n"
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
    )

    _set_paral_config()

    monitor = TorchTrainingMonitor(ConfigPath.RUNTIME_METRICS)
    if config.auto_tunning:
        monitor.start()
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
        RendezvousName.ELASTIC_TRAINING,
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
        redirects=config.redirects,
        tee=config.tee,
        master_addr=master_addr,
    )

    agent = ElasticTrainingAgent(
        node_rank=node_rank,
        config=config,
        entrypoint=entrypoint,
        spec=spec,
        start_method=config.start_method,
        log_dir=config.log_dir,
    )

    shutdown_rdzv = True
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
    except Exception:
        events.record(agent.get_event_failed())
        raise
    finally:
        if shutdown_rdzv:
            spec.rdzv_handler.shutdown()
        agent.stop_executor()
        monitor.stop()


class NetworkCheckElasticAgent(ElasticTrainingAgent):
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
    ):
        super().__init__(
            node_rank,
            config,
            entrypoint,
            spec,
            start_method,
            exit_barrier_timeout,
            log_dir,
        )
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="network_check_")
        self._check_round = 2
        self._config: ElasticLaunchConfig = config

    def run(self, role: str = DEFAULT_ROLE) -> bool:
        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            f"[{role}] starting workers for entrypoint: "
            f"{spec.get_entrypoint_name()}"
        )
        success = False
        fault_nodes = []
        stragglers = []
        for i in range(self._check_round):
            result, elapsed_time = self._run_network_check()
            elapsed_time = round(elapsed_time, 3)
            logger.info(
                f"Network check time of round {i} is {elapsed_time}"
                f" and succeed is {result}."
            )
            status = NodeStatus.SUCCEEDED if result else NodeStatus.FAILED
            self._client.report_network_status(
                self._node_rank,
                status,
                elapsed_time,
            )
            success = success or result
            fault_nodes = self._client.check_fault_node()
            stragglers = self._client.check_straggler()
            logger.info(
                f"Fault nodes are: {fault_nodes} "
                f" and stragglers are: {stragglers}."
            )
            self._stop_workers(self._worker_group)
            if fault_nodes or stragglers:
                total_worker_num = len(self._client.get_running_nodes())
                if total_worker_num <= 3:
                    # If the number of nodes <= 3, we cannot determine which
                    # node if fault because there is no normal node in the job
                    # to execute allgather tasks with the two nodes.
                    logger.error("Network check needs at least 4 nodes.")
                    raise RuntimeError("The node network is breakdown.")
                else:
                    # Run the next round check to detect the fault node.
                    time.sleep(3)
                    continue
            else:
                return True
        if self._node_rank in fault_nodes:
            self._client.report_failures(
                NodeErrorMessage.NETWORKER_ERROR,
                level=TrainingMsgLevel.NODE_ERROR,
            )
            raise RuntimeError("The node network is breakdown.")
        elif self._config.exclude_straggler and self._node_rank in stragglers:
            raise RuntimeError("The node is a straggler and exits.")
        return True

    def _run_network_check(self, monitor_interval=3, timeout=300):
        self._initialize_workers(self._worker_group)
        start = time.time()
        succeed = False
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
            elif state == WorkerState.SUCCEEDED:
                succeed = True
                break
            else:
                break

        if succeed:
            elapsed_time = self._get_network_check_time()
        else:
            elapsed_time = 3600
        return succeed, elapsed_time

    def _get_network_check_time(self):
        root = ConfigPath.NETWORK_CHECK_DATA_DIR
        elapsed_time = 0
        if not os.path.exists(root):
            return elapsed_time
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            with open(path, "r") as f:
                data = f.read()
                if not data:
                    continue
                data = json.loads(data)
                elapsed_time = max(elapsed_time, data.get("time", 0))
        return elapsed_time


def network_check(
    config: ElasticLaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> bool:
    config = copy.deepcopy(config)
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
        f"Starting elastic_operator with launch configs:\n"
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
    )

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
        RendezvousName.NETWORK_CHECK,
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
        max_restarts=0,
        monitor_interval=config.monitor_interval,
        master_addr=master_addr,
    )

    agent = NetworkCheckElasticAgent(
        node_rank=node_rank,
        config=config,
        entrypoint=entrypoint,
        spec=spec,
        start_method=config.start_method,
    )

    metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
    result = agent.run()
    logger.info("Network check result is %s", result)
    return result


def run_network_check(config, entrypoint):
    cmd_args = ["-m", "dlrover.trainer.torch.run_network_check"]
    for _ in range(2):
        # If network fails because other abnormal node, We
        # will retry to check network after the new node is starting.
        # DLRover will replace the abnormal node with a new node.
        success = network_check(
            config=config, entrypoint=entrypoint, args=cmd_args
        )
        if success:
            logger.info("Network check pass.")
            return success
        else:
            logger.error(
                "Network of the cluster is not available "
                "because of abnormal node."
            )
    return success
