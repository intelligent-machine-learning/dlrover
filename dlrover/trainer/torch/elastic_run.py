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

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import torch.distributed.elastic.rendezvous.registry as rdzv_registry
import torch.distributed.elastic.timer as timer
from torch.distributed.elastic import events, metrics
from torch.distributed.elastic.agent.server.api import (
    DEFAULT_ROLE,
    RunResult,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
)
from torch.distributed.elastic.metrics import put_metric
from torch.distributed.elastic.multiprocessing import PContext, SignalException
from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
    record,
)
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.api import RendezvousHandler
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import (
    LaunchConfig,
    _get_addr_and_port,
    _get_entrypoint_name,
)
from torch.distributed.run import config_from_args, parse_args

from dlrover.python.elastic_agent.master_client import GlobalMasterClient

__all__ = ["LaunchConfig", "elastic_launch", "launch_agent"]


logger = get_logger()


@dataclass
class ProcessError:
    local_rank: int
    exitcode: int
    message: str
    datetime: Any


class DLRoverElasticAgent(LocalElasticAgent):
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
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        self._log_dir = log_dir
        self._worker_watchdog: Optional[timer.FileTimerServer] = None
        self._reamining_fo_count: int = self._remaining_restarts
        self._client = GlobalMasterClient.MASTER_CLIENT

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
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(
                f"workers.{role}.remaining_restarts", self._remaining_restarts
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
                if self._reamining_fo_count > 0:
                    logger.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_restarts}/{spec.max_restarts}"
                        f" attempts left; will restart worker group"
                    )
                    self._reamining_fo_count -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                if self._membership_changed(role, rdzv_handler):
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    def _report_failure_to_master(self, failures: Dict[int, ProcessFailure]):
        errors = {}
        for rank, failure in failures.items():
            dt = str(datetime.utcfromtimestamp(int(failure.timestamp)))
            error = ProcessError(
                failure.local_rank, failure.exitcode, failure.message, dt
            )
            errors[rank] = error.__dict__
        error_data = json.dumps(errors)
        self._client.report_failures(error_data)

    def _restart_workers(self, worker_group: WorkerGroup):
        self._remaining_restarts -= 1
        super()._restart_workers(worker_group)

    def should_shutdown_rdzv(self):
        return self._reamining_fo_count == 0

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

    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        redirects=config.redirects,
        tee=config.tee,
        master_addr=master_addr,
        master_port=master_port,
        local_addr=config.local_addr,
    )

    agent = DLRoverElasticAgent(
        spec=spec, start_method=config.start_method, log_dir=config.log_dir
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
        if not agent.should_shutdown_rdzv():
            shutdown_rdzv = False
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


class elastic_launch:
    """
    Launches an torchelastic agent on the container
    that invoked the entrypoint.

        1. Pass the ``entrypoint`` arguments as non ``kwargs``
            (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker's output mapped
           by their respective global rank.

    Usage

    ::

    def worker_fn(foo):
        # ...

    def main():
        # entrypoint is a function.
        outputs = elastic_launch(LaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = elastic_launch(LaunchConfig, "script.py")(args)
        outputs = elastic_launch(LaunchConfig, "python")("script.py")
    """

    def __init__(
        self,
        config: LaunchConfig,
        entrypoint: Union[Callable, str, None],
    ):
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))


def run(args):
    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:29400"
        args.rdzv_id = str(uuid.uuid4())
        logger.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv-backend={args.rdzv_backend} "
            f"--rdzv-endpoint={args.rdzv_endpoint} "
            f"--rdzv-id={args.rdzv_id}\n"
            f"**************************************\n"
        )

    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


@record
def main(args=None):
    args = parse_args(args)
    run(args)
