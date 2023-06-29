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

import os
import socket
import tempfile
import time
import uuid
from typing import Any, Callable, List, Optional, Union

import torch.distributed.elastic.timer as timer
from torch.distributed.elastic import metrics
from torch.distributed.elastic.agent.server.api import (
    DEFAULT_ROLE,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
    _get_fq_hostname,
)
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import PContext
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.launcher.api import LaunchConfig, _get_entrypoint_name

from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
from dlrover.python.elastic_agent.torch.training import (
    ElasticTrainingAgent,
    MasterRendezvousHandler,
)


class NcclCheckElasticAgent(ElasticTrainingAgent):
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
        node_id,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        self._worker_watchdog: Optional[timer.FileTimerServer] = None
        self._reamining_fo_count: int = self._remaining_restarts
        self._node_id = node_id
        self._client = GlobalMasterClient.MASTER_CLIENT

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    @prof
    def _initialize_workers(
        self, worker_group: WorkerGroup, round: int
    ) -> None:
        r"""
        Starts a fresh set of workers for the worker_group.
        Essentially a rendezvous followed by a start_workers.

        The caller should first call ``_stop_workers()`` to stop running
        workers prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role
        logger.info(f"[{role}] Rendezvous'ing worker group")

        # TODO after stopping workers, wait at least monitor_interval*2 for
        # workers on different nodes to fail on a collective op before waiting
        # on the rdzv barrier, this way we ensure that nodes enter rdzv
        # at around the same time and reduce false positive rdzv timeout errors
        self._rendezvous(worker_group, round)

        logger.info(f"[{role}] Starting worker group")
        worker_ids = self._start_workers(worker_group)
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id

        worker_group.state = WorkerState.HEALTHY

    def run(self, role: str = DEFAULT_ROLE) -> bool:
        spec = self._worker_group.spec
        role = spec.role

        logger.info(
            f"[{role}] starting workers for entrypoint: "
            f"{spec.get_entrypoint_name()}"
        )
        for i in range(3):
            self._run_nccl_check(i, spec.monitor_interval)
            success = self._client.nccl_check_success(self._node_id)
            if success:
                return success
        return False

    def _run_nccl_check(self, round, monitor_interval):
        self._initialize_workers(self._worker_group, round)

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            if state == WorkerState.SUCCEEDED:
                self._client.report_nccl_check_result(self._node_id, True)
                break
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                self._client.report_nccl_check_result(self._node_id, False)
                break


def ncck_check(
    config: LaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> bool:
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning(
            f"config has no run_id, generated a random run_id: {run_id}"
        )
        config.run_id = run_id

    entrypoint_name = _get_entrypoint_name(entrypoint, args)
    node_id = int(os.getenv(NodeEnv.WORKER_ID, 0))

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
        "nccl-check",
        node_id,
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
        master_addr=master_addr,
    )

    agent = NcclCheckElasticAgent(
        node_id=node_id,
        spec=spec,
        start_method=config.start_method,
        log_dir=config.log_dir,
    )

    try:
        metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
        for i in range(3):
            result = agent.run()
            if result:
                return True
            time.sleep(15)
    finally:
        return True
