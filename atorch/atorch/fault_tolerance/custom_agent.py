import os
import shutil
import socket
import time
from contextlib import closing

from torch.distributed.elastic.agent.server.api import WorkerState, _get_fq_hostname, _get_socket_with_port
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger

from atorch.fault_tolerance.hanging_detector import RelaunchStatus

log = get_logger()


class LocalDetectHangingAgent(LocalElasticAgent):
    def __init__(
        self,
        spec,
        start_method="spawn",
        exit_barrier_timeout=300,
        log_dir=None,
        rdzv_params=None,
    ):
        super().__init__(spec, start_method, exit_barrier_timeout, log_dir)
        self.rdzv_params = rdzv_params
        self.node_world_size = self.rdzv_params.max_nodes
        self.node_rank = self.rdzv_params.config.get("node_rank")
        if self.node_rank is None:
            self.node_rank = os.getenv("RANK")
        if self.node_rank is None:
            self.node_rank = "0"

    @staticmethod
    def _set_master_addr_port(store, master_addr, master_port, local_dir=None):
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

        if master_addr is None:
            if local_dir is not None:
                master_addr = local_dir
            else:
                master_addr = os.getenv("POD_IP", socket.gethostbyname(_get_fq_hostname()))

        store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

    def _invoke_run(self, role):
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        log.info(f"[{role}] starting workers for entrypoint: {spec.get_entrypoint_name()}")

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        # self._store is a PrefixStore. Here we use he underlying TCPStore of the self._store.
        underlying_store = rdzv_handler._state_holder._backend._store

        worker_world_size = 0
        workers = self._worker_group.workers
        if workers:
            worker_world_size = workers[0].world_size
        relaunch_status = None
        if worker_world_size > 0:
            node_rank = int(self.node_rank)
            node_world_size = int(self.node_world_size)
            relaunch_status = RelaunchStatus(
                "agent",
                worker_world_size=worker_world_size,
                agent_rank=node_rank,
                agent_world_size=node_world_size,
                store=underlying_store,
            )
        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                log.info(
                    f"[{role}] worker group successfully finished."
                    f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish."
                )
                self._exit_barrier()
                return run_result
            elif relaunch_status is not None and relaunch_status.should_relaunch():
                self._restart_workers(self._worker_group)
                relaunch_status.step()
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if relaunch_status is not None and relaunch_status.should_relaunch():
                    self._restart_workers(self._worker_group)
                    relaunch_status.step()
                elif self._remaining_restarts > 0:
                    log.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                        f" will restart worker group"
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    log.info(
                        f"[{role}] Detected {num_nodes_waiting} "
                        f"new nodes from group_rank={group_rank}; "
                        f"will restart worker group"
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    @prof
    def _start_workers(self, worker_group):
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"
        agent_master_addr, agent_master_port = self.rdzv_params.endpoint.split(":")

        args = {}
        envs = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": str(1),
                "TORCHELASTIC_AGENT_MASTER_ADDR": agent_master_addr,
                "TORCHELASTIC_AGENT_MASTER_PORT": agent_master_port,
                "NODE_RANK": self.node_rank,
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        # scaling events do not count towards restarts (gets same attempt #)
        # remove existing log dir if this restart is due to a scaling event
        attempt_log_dir = os.path.join(self._log_dir, f"attempt_{restart_count}")
        shutil.rmtree(attempt_log_dir, ignore_errors=True)
        os.makedirs(attempt_log_dir)

        assert spec.entrypoint is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
        )

        return self._pcontext.pids()
