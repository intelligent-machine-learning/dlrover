import os
import threading
import time

import torch.distributed as dist

from atorch.distributed.distributed import rank as get_rank


class RelaunchStatus(object):
    """
    RelaunchStatus is responsible for:
    - Create connections between agents and workers.
    - Checkout whether agent should restart its workers.

    Args:
        relaunch_id: If a worker's relaunch_id is larger than any of agent's relaunch_id, restart workers.
        role: 'agent' or 'worker'.
    """

    def __init__(
        self, role, worker_rank=None, worker_world_size=None, agent_rank=None, agent_world_size=None, store=None
    ):
        self.relaunch_id = 0
        self.role = role
        self.worker_rank = worker_rank
        self.worker_world_size = worker_world_size
        self.agent_rank = agent_rank
        self.agent_world_size = agent_world_size
        self.store_prefix = "relaunch_status"
        self.store = store
        self.initialized = True
        self.agent_key = None
        self.worker_key = None
        self.init_env()

    def init_env(self):
        if self.role not in ("agent", "worker"):
            raise ValueError("Argument 'role' only support agent or worker")
        if self.role == "worker":
            host_name = os.getenv("TORCHELASTIC_AGENT_MASTER_ADDR", None)
            if host_name is None:
                self.initialized = False
                return

            port = int(os.getenv("TORCHELASTIC_AGENT_MASTER_PORT"))
            self.agent_rank = int(os.getenv("NODE_RANK"))
            self.agent_key = f"agent_{self.agent_rank}"
            self.worker_key = f"{self.role}_{self.worker_rank}"
            tcp_store = dist.TCPStore(host_name, port, is_master=False)
            self.store = dist.PrefixStore(self.store_prefix, tcp_store)
            self.relaunch_id = int(self.store.get(self.agent_key).decode("UTF-8"))
        else:  # "agent"
            assert self.store is not None
            assert self.agent_rank is not None
            self.agent_key = f"{self.role}_{self.agent_rank}"
            self.store = dist.PrefixStore(self.store_prefix, self.store)
            # Every agent initialize its own relaunch_id
            self.store.set(self.agent_key, str(self.relaunch_id))
            if self.agent_rank == 0:
                # Agent0 initialize all workers' relaunch_id.
                for rank in range(self.worker_world_size):
                    self.store.set(f"worker_{rank}", str(self.relaunch_id))

    def step(self):
        if self.role == "agent":
            self.relaunch_id += 1
            self.store.set(self.agent_key, str(self.relaunch_id))

    def set_relaunch(self):
        if self.role == "worker":
            self.relaunch_id += 1
            self.store.set(self.worker_key, str(self.relaunch_id))

    def should_relaunch(self):
        if self.role == "agent":
            workers_relaunch_id = [
                int(self.store.get(f"worker_{worker_rank}").decode("UTF-8"))
                for worker_rank in range(self.worker_world_size)
            ]
            max_worker_relaunch_id = max(workers_relaunch_id)
            should = max_worker_relaunch_id > self.relaunch_id
            return should


class HangingDetector(object):
    """
    Detect hanging during distributed training and restart all worker processes when hanging. The training
    script should be launched by 'python3 -m atorch.distributed.run --relaunch_on_hanging train_script.py'

    Args:
        timeout: Timeout for hanging. Default value is 300 seconds.
        monitor_interval: Check out if hanging every `monitor_interval` seconds.
        rank: The rank of a worker. Can get from atorch.rank() or torch.distributed.get_rank().
    """

    def __init__(self, timeout=300, monitor_interval=15, rank=None):
        self._last_report_time = None
        self.timeout = timeout
        self.monitor_interval = monitor_interval
        self.is_initialized = False
        self.is_running = False
        self.worker_rank = rank if rank is not None else get_rank()
        self.relaunch_status = RelaunchStatus("worker", worker_rank=self.worker_rank)
        self.thread = None
        self.enabled = None
        if not self.relaunch_status.initialized:
            raise RuntimeError(
                "Initialize HangingDetector failed. Launch training script by 'python3 -m atorch.distributed.run "
                "--relaunch_on_hanging train_script.py'"
            )

    def start(self):
        """Start Detecting. The method should be called only once."""
        if self.enabled is None:
            self.enabled = True
        if not self.is_running:
            self._last_report_time = time.time()
            self.is_running = True
            if self.enabled is True and self.thread is None:
                self.thread = threading.Thread(target=self.detector_timeout, daemon=True)
                self.thread.start()

    def stop(self, finalize=False):
        """Stop Detecting.
        Args:
            If finalize is True, HangingDetector will be disabled.
        """
        self.is_running = False
        if finalize:
            self.enabled = False

    def report_normal(self):
        """During training, call this method at intervals or every training steps."""
        self._last_report_time = time.time()

    def detector_timeout(self):
        while True:
            if not self.enabled:
                break
            time.sleep(self.monitor_interval)
            now = time.time()
            if self.is_running and now - self._last_report_time > self.timeout:
                self.relaunch_status.set_relaunch()
                break
