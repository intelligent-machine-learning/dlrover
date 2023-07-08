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
import socket
import tempfile
import time
import uuid
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

from dlrover.python.common.constants import (
    NodeEnv,
    NodeErrorMessage,
    NodeStatus,
    RendezvousName,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore

__all__ = ["launch_agent"]


@dataclass
class ProcessError:
    local_rank: int
    exitcode: int
    message: str
    datetime: Any


class MasterRendezvousHandler(RendezvousHandler):
    def __init__(self, name, rank_id, rdzv_params: RendezvousParameters):
        self._name = name
        self._rank_id = rank_id
        self._rdzv_params = rdzv_params
        self.join_timeout = int(rdzv_params.get("join_timeout", 600))
        self._client = GlobalMasterClient.MASTER_CLIENT
        self._store = MasterKVStore(self._name, timedelta(seconds=60))
        lastcall_timeout = int(rdzv_params.get("lastcall_timeout", 60))
        if self._rank_id == 0:
            self._client.report_rdzv_params(
                rdzv_params.min_nodes,
                rdzv_params.max_nodes,
                lastcall_timeout,
            )

    def get_backend(self) -> str:
        return "dlrover-master"

    def is_closed(self) -> bool:
        return False

    def set_closed(self):
        """Marks the rendezvous as closed."""
        pass

    def join_rendezvous(self, local_world_size):
        """The node join a rendezvous by sending its
        ID and local world size.
        """
        round = self._client.join_rendezvous(
            self._rank_id, local_world_size, rdzv_name=self._name
        )
        return round

    def next_rendezvous(self, round):
        """The handler will peroidically query the world from the master until
        the world is not empty. The world is a dictionary like
        like {0: 8, 1: 8, 2: 8} where the key is the node ID and the value is
        the local world size. The handler can get its rank by the position
        of it node ID in the world.
        """
        start_join = time.time()
        node_name = os.getenv("POD_NAME", "")
        msg = (
            f"The node node_name attempts to join the next round of the "
            f"rendezvous '{self._name}' with timeout {self.join_timeout}."
        )
        logger.info(msg)
        while True:
            group, world = self._client.get_comm_world(
                self._name, self._rank_id
            )
            world = dict(sorted(world.items()))
            if world:
                break
            if time.time() - start_join > self.join_timeout:
                raise TimeoutError(
                    f"Timeout {self.join_timeout}s to complete next rendezous."
                )
            time.sleep(3)
        rank = list(world.keys()).index(self._rank_id)
        world_size = len(world)
        logger.info(
            f"The node{node_name} has joined round {round} of "
            f"the {self._name} rendezvous as rank {rank} in a world of size "
            f"{world_size}."
        )
        store = self._get_store(round, group)
        return store, world

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
        rank_id,
        config,
        entrypoint,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._rank_id = rank_id
        self._config = config
        self._entrypoint = entrypoint
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        self._worker_watchdog: Optional[timer.FileTimerServer] = None
        self._restart_count = 0
        self._remaining_failovers = self._remaining_restarts
        self._client = GlobalMasterClient.MASTER_CLIENT

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""
        Runs rendezvous for the workers specified by worker spec.
        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """

        spec = worker_group.spec
        round = spec.rdzv_handler.join_rendezvous(spec.local_world_size)
        store, world = spec.rdzv_handler.next_rendezvous(round)
        self._store = store
        group_world_size = len(world)
        group_rank = list(world.keys()).index(self._rank_id)

        workers = self._assign_worker_ranks(self._rank_id, world, spec)
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size

        if group_rank == 0:
            self._set_master_addr_port(
                store,
                spec.master_addr,
                spec.master_port,
                spec.local_addr,
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
        if self._config.network_check and self._restart_count == 0:
            run_network_check(self._config, self._entrypoint)
        super()._initialize_workers(worker_group)

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

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
            run_result: RunResult = self._monitor_workers(self._worker_group)
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
                self._report_failure_to_master(run_result.failures)
                has_fatal_error = self._has_fatal_error(run_result)
                if not has_fatal_error and self._remaining_failovers > 0:
                    logger.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_failovers}/{spec.max_restarts}"
                        f" attempts left; will restart worker group"
                    )
                    self._remaining_failovers -= 1
                    self._restart_workers(self._worker_group)
                else:
                    logger.info("Cannot restart workers with fatal error.")
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                if self._membership_changed(role, rdzv_handler):
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    def _has_fatal_error(self, run_result: RunResult):
        """The error with exitcode 1 is the Python exception and we cannot
        recover it by restarting workers."""
        for pfailure in run_result.failures.values():
            if pfailure.exitcode == 1:
                return True
        return False

    def _report_failure_to_master(self, failures: Dict[int, ProcessFailure]):
        errors = {}
        for rank, failure in failures.items():
            dt = str(datetime.fromtimestamp(int(failure.timestamp)))
            error = ProcessError(
                failure.local_rank, failure.exitcode, failure.message, dt
            )
            errors[rank] = error.__dict__
        error_data = json.dumps(errors)
        self._client.report_failures(error_data, self._restart_count)

    def _restart_workers(self, worker_group: WorkerGroup):
        self._restart_count += 1
        self._remaining_restarts -= 1
        super()._restart_workers(worker_group)

    def _membership_changed(self, role, rdzv_handler: RendezvousHandler):
        # Timeout may happen when to query TCPStore.
        try:
            num_nodes_waiting = rdzv_handler.num_nodes_waiting()
        except Exception as e:
            logger.warning("Fail to call num_node_waiting.", e)
            num_nodes_waiting = 0

        group_rank = self._worker_group.group_rank
        if num_nodes_waiting > 0:
            logger.info(
                f"[{role}] Detected {num_nodes_waiting} "
                f"new nodes from group_rank={group_rank}; "
                f"will restart worker group"
            )
            return True
        return False


def launch_agent(
    config: LaunchConfig,
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
    rank_id = int(os.getenv(NodeEnv.WORKER_RANK, 0))

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
        local_addr=config.local_addr,
        **config.rdzv_configs,
    )

    master_addr = os.environ.get(
        "MY_POD_IP", socket.gethostbyname(_get_fq_hostname())
    )

    rdzv_handler = MasterRendezvousHandler(
        RendezvousName.ELASTIC_TRAINING,
        rank_id,
        rdzv_parameters,
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
        local_addr=config.local_addr,
    )

    agent = ElasticTrainingAgent(
        rank_id=rank_id,
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


class NcclCheckElasticAgent(ElasticTrainingAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers. This agent will run 3 round allgather
    to check network. We show the detail with 4 nodes to check network.
    Round 1: all nodes join a communication world {0:8, 1:8, 2:8, 3:8}
        where the key is the node id and the value is the local world size
        of the node. The check passes if allgather of all nodes is succeed.
        Otherwise, the round 2 starts.
    Round 2: the manager splits nodes into groups and each group contains
        two nodes, like [{0:8, 1:8},{2:8, 3:8}]. The node in each group will
        execute allgather independently and report its result to the manager.
        For example, the result is {0:False, 1:False, 2:True, 3:True}.
    Round 3: the manager will group the abnormal node with a normal node like
        [{0:8, 2:8}, {1:8, 2:8}]. Then, the node executes allgather again.
        If the result is {0:True, 1:False, 2:False, 3:True}, the network of
        node-1 if not available.
    """

    def __init__(
        self,
        rank_id,
        config,
        entrypoint,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(
            rank_id,
            config,
            entrypoint,
            spec,
            start_method,
            exit_barrier_timeout,
            log_dir,
        )
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="network_check_")
        self._max_check_round = 3

    def run(self, role: str = DEFAULT_ROLE) -> bool:
        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            f"[{role}] starting workers for entrypoint: "
            f"{spec.get_entrypoint_name()}"
        )
        success = False
        for i in range(self._max_check_round):
            result = self._run_network_check(spec.monitor_interval)
            logger.info(f"Network check round {i} is {result}")
            status = NodeStatus.SUCCEEDED if result else NodeStatus.FAILED
            self._client.report_node_status(self._rank_id, status)
            success = success or result
            network_ready = self._client.network_check_success()
            self._stop_workers(self._worker_group)
            if network_ready:
                return True
            elif i == 0 and self._worker_group.group_world_size <= 2:
                logger.error(
                    "Fail to check network when there are only 2 nodes."
                )
                raise RuntimeError("The node network is breakdown.")
            time.sleep(1)
        if not success:
            self._client.report_failures(NodeErrorMessage.NETWORKER_ERROR)
            raise RuntimeError("The node network is breakdown.")
        return False

    def _run_network_check(self, monitor_interval, timeout=300):
        self._initialize_workers(self._worker_group)
        start = time.time()
        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state
            if state == WorkerState.HEALTHY:
                if time.time() - start > timeout:
                    logger.error(f"Timeout {timeout} to check network.")
                    return False
                continue
            return state == WorkerState.SUCCEEDED


def network_check(
    config: LaunchConfig,
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
    rank_id = int(os.getenv(NodeEnv.WORKER_RANK, 0))

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
        local_addr=config.local_addr,
        **config.rdzv_configs,
    )

    master_addr = os.environ.get(
        "MY_POD_IP", socket.gethostbyname(_get_fq_hostname())
    )
    rdzv_handler = MasterRendezvousHandler(
        RendezvousName.NETWORK_CHECK,
        rank_id,
        rdzv_parameters,
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

    agent = NcclCheckElasticAgent(
        rank_id=rank_id,
        config=config,
        entrypoint=entrypoint,
        spec=spec,
        start_method=config.start_method,
        log_dir=config.log_dir,
    )

    metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
    result = agent.run()
    logger.info("Network check result is %s", result)
    return result


def run_network_check(config, entrypoint):
    cmd_args = ["-m", "dlrover.trainer.torch.run_network_check"]
    for _ in range(config.max_restarts):
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
