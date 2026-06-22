# Copyright 2025 The DLRover Authors. All rights reserved.
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

"""Helpers for spinning up a DLRover master subprocess for integration tests.

Two master types are supported:

``master_type="local"``  (default)
    Starts ``dlrover.python.master.main`` with ``--platform local``.
    Uses ``LocalJobMaster`` / ``LocalJobManager``.
    Identical to the standalone mode started by ``elastic_run.py``.

``master_type="dist"``
    Starts ``dlrover.python.testing.master.sim_master_main``.
    Uses ``DistributedJobMaster`` / ``DistributedJobManager`` with
    no-op stubs in place of the Kubernetes dependencies, so the full
    production code path (failure handling, rendezvous, OOM behaviour)
    is exercised without a real cluster.

In both cases fake agents connect via gRPC using ``GrpcMasterClient``,
just as real ``ElasticTrainingAgent`` instances would from worker nodes.

Usage::

    # Local master (LocalJobMaster)
    with MasterContext(num_workers=4) as ctx:
        agents = make_agents(ctx.addr, num_agents=4)
        results = run_agents(agents)

    # Distributed master (DistributedJobMaster)
    with MasterContext(num_workers=4, master_type="dist") as ctx:
        agents = make_agents(ctx.addr, num_agents=4)
        results = run_agents(agents)
"""

import subprocess
import sys
import threading
import time
from typing import Optional

from dlrover.python.common.comm import addr_connected
from dlrover.python.common.log import default_logger as logger
from dlrover.python.util.common_util import find_free_port


class MasterContext:
    """Context manager that starts a DLRover master as a subprocess.

    The master process is terminated when the ``with`` block exits.

    Args:
        master_type: ``"local"`` for ``LocalJobMaster`` (via
            ``dlrover.python.master.main --platform local``), or
            ``"dist"`` for ``DistributedJobMaster`` (via
            ``dlrover.python.testing.master.sim_master_main``).
        num_workers: Number of worker agents the master expects.
        max_relaunch_count: How many times to relaunch a failed worker.
        port: gRPC port to listen on.  Auto-selected if ``None``.
        startup_timeout: Seconds to wait for the master to become ready.
        job_name: Job name passed to the master process.
        namespace: Kubernetes namespace passed to the master process.

    Attributes:
        addr: ``"host:port"`` string agents should connect to.
        master_proc: The ``subprocess.Popen`` object for the master process.
    """

    def __init__(
        self,
        master_type: str = "local",
        num_workers: int = 4,
        max_relaunch_count: int = 3,
        port: Optional[int] = None,
        startup_timeout: float = 15.0,
        job_name: str = "sim-test",
        namespace: str = "default",
    ):
        if master_type not in ("local", "dist"):
            raise ValueError(
                f"master_type must be 'local' or 'dist', got {master_type!r}"
            )
        self._master_type = master_type
        self._num_workers = num_workers
        self._max_relaunch_count = max_relaunch_count
        self._port = port or find_free_port()
        self._startup_timeout = startup_timeout
        self._job_name = job_name
        self._namespace = namespace

        self.addr: str = f"127.0.0.1:{self._port}"
        self.master_proc: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MasterContext":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Explicit start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the master subprocess and wait until it is ready."""
        cmd = self._build_cmd()
        logger.info(f"Starting {self._master_type} master: {' '.join(cmd)}")
        self.master_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._log_thread = threading.Thread(
            target=self._forward_logs, daemon=True
        )
        self._log_thread.start()
        self._wait_for_ready()
        logger.info(
            f"Master ({self._master_type}) ready at {self.addr} "
            f"(pid={self.master_proc.pid})"
        )

    def _forward_logs(self) -> None:
        """Daemon thread: read master subprocess stdout and re-log each line."""
        for raw in self.master_proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                logger.info(f"[master] {line}")

    def stop(self) -> None:
        """Terminate the master subprocess."""
        if self.master_proc is not None and self.master_proc.poll() is None:
            logger.info(
                f"Stopping master subprocess (pid={self.master_proc.pid})"
            )
            self.master_proc.terminate()
            try:
                self.master_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.master_proc.kill()
                self.master_proc.wait()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cmd(self):
        common = [
            "--port",
            str(self._port),
            "--node_num",
            str(self._num_workers),
            "--job_name",
            self._job_name,
            "--namespace",
            self._namespace,
        ]
        if self._master_type == "local":
            return [
                sys.executable,
                "-u",
                "-m",
                "dlrover.python.master.main",
                "--platform",
                "local",
            ] + common
        else:  # "dist"
            return [
                sys.executable,
                "-u",
                "-m",
                "dlrover.python.testing.master.sim_master_main",
                "--max_relaunch_count",
                str(self._max_relaunch_count),
            ] + common

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self.master_proc.poll() is not None:
                # Give the log thread a moment to flush any remaining output.
                if self._log_thread is not None:
                    self._log_thread.join(timeout=1.0)
                raise RuntimeError(
                    f"Master process exited early "
                    f"(rc={self.master_proc.returncode})"
                )
            if addr_connected(self.addr):
                return
            time.sleep(0.2)
        self.stop()
        raise TimeoutError(
            f"Master at {self.addr} did not become ready within "
            f"{self._startup_timeout}s"
        )
