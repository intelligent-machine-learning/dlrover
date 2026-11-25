#!/usr/bin/env python3
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
"""Run one node of the two-node TP2 x PP2 x DP2 demo.

Start this script once on each node with the same shared ``--work-dir``.
Node 0 coordinates recovery.  The node that owns the failed global rank
starts the replacement worker; healthy workers stay in their existing
processes.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from recovery_protocol import (
    RecoveryCommand,
    atomic_write_json,
    fresh_reconfigure_uuid,
    load_recovery_command,
    read_json,
)


WORLD_SIZE = 8
NODE_COUNT = 2
GPUS_PER_NODE = 4
TP_SIZE = 2
PP_SIZE = 2
DP_SIZE = 2
TRAIN_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "tp2_pp2_dp2"
    / "hybrid_moe_train.py"
)


def _coordinates(rank: int) -> str:
    tp_rank = rank % TP_SIZE
    pp_rank = (rank // TP_SIZE) % PP_SIZE
    dp_rank = rank // (TP_SIZE * PP_SIZE)
    return f"TP{tp_rank}-PP{pp_rank}-DP{dp_rank}"


def _node_ranks(node_rank: int) -> Tuple[int, ...]:
    start = node_rank * GPUS_PER_NODE
    return tuple(range(start, start + GPUS_PER_NODE))


def _wait_for_json(
    path: Path,
    timeout_seconds: float,
    description: str,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = read_json(path)
        if value is not None:
            return value
        time.sleep(0.2)
    raise TimeoutError(
        f"timed out waiting for {description}: {path}"
    )


def _wait_for_json_matching(
    path: Path,
    timeout_seconds: float,
    description: str,
    predicate: Callable[[Dict[str, Any]], bool],
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = read_json(path)
        if value is not None and predicate(value):
            return value
        time.sleep(0.2)
    raise TimeoutError(
        f"timed out waiting for {description}: {path}"
    )


def _read_all_failures(
    failure_dir: Path,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for path in sorted(failure_dir.glob("rank_*.json")):
        value = read_json(path)
        if value is not None:
            failures.append(value)
    return failures


class NodeAgent:
    """Supervise the four local ranks and participate in recovery."""

    def __init__(
        self,
        args: argparse.Namespace,
        work_dir: Path,
    ) -> None:
        self.args = args
        self.node_rank = args.node_rank
        self.work_dir = work_dir
        self.ranks = _node_ranks(self.node_rank)
        self.state_path = (
            self.work_dir
            / "agents"
            / f"node_{self.node_rank}.json"
        )
        self.failure_dir = self.work_dir / "agents" / "failures"
        self.command_path = (
            self.work_dir / "recovery" / "command.json"
        )
        self.processes: Dict[int, subprocess.Popen] = {}
        self.initial_pids: Dict[int, int] = {}
        self.replacement_pids: Dict[int, int] = {}
        self.reported_failures: Set[Tuple[int, int]] = set()
        self.reported_recoveries: Set[int] = set()
        self.peer_initial_pids: Dict[int, int] = {}
        self.recovery_started = False
        self.last_command_epoch = 0

    def _log(self, message: str) -> None:
        print(
            f"[node-agent {self.node_rank}] {message}",
            flush=True,
        )

    def _write_state(
        self,
        phase: str,
        **extra: Any,
    ) -> None:
        state: Dict[str, Any] = {
            "node_rank": self.node_rank,
            "node_count": NODE_COUNT,
            "hostname": socket.gethostname(),
            "agent_pid": os.getpid(),
            "work_dir": str(self.work_dir),
            "phase": phase,
            "local_ranks": list(self.ranks),
            "initial_pids": {
                str(rank): pid
                for rank, pid in sorted(self.initial_pids.items())
            },
            "replacement_pids": {
                str(rank): pid
                for rank, pid in sorted(
                    self.replacement_pids.items()
                )
            },
            "timestamp_unix_ms": int(time.time() * 1000),
        }
        state.update(extra)
        atomic_write_json(self.state_path, state)

    def _initialize_run(self) -> None:
        if self.node_rank != 0:
            raise RuntimeError(
                "only node 0 may initialize a run directory"
            )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if any(self.work_dir.iterdir()):
            raise ValueError(
                "work directory is not empty: "
                f"{self.work_dir}; use a new shared directory"
            )
        metadata = {
            "format": "dlrover-in-place-two-node",
            "work_dir": str(self.work_dir),
            "world_size": WORLD_SIZE,
            "node_count": NODE_COUNT,
            "gpus_per_node": GPUS_PER_NODE,
            "tp_size": TP_SIZE,
            "pp_size": PP_SIZE,
            "dp_size": DP_SIZE,
            "ep_size": 1,
            "created_unix_ms": int(time.time() * 1000),
        }
        atomic_write_json(self.work_dir / "run.json", metadata)
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "recovery").mkdir(
            parents=True,
            exist_ok=True,
        )

    def _wait_for_run(self) -> None:
        metadata = _wait_for_json(
            self.work_dir / "run.json",
            self.args.job_timeout_seconds,
            "node 0 to initialize the shared work directory",
        )
        if metadata.get("format") != "dlrover-in-place-two-node":
            raise ValueError("unexpected run format")
        if metadata.get("work_dir") != str(self.work_dir):
            raise ValueError(
                "both nodes must use the same absolute --work-dir"
            )
        if int(metadata.get("world_size", 0)) != WORLD_SIZE:
            raise ValueError("run world size does not match")
        if int(metadata.get("node_count", 0)) != NODE_COUNT:
            raise ValueError("run node count does not match")
        if int(metadata.get("gpus_per_node", 0)) != GPUS_PER_NODE:
            raise ValueError("run GPUs per node does not match")

    def _launch_worker(
        self,
        rank: int,
        replacement: bool,
    ) -> subprocess.Popen:
        local_rank = rank % GPUS_PER_NODE
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(WORLD_SIZE),
                "LOCAL_WORLD_SIZE": str(GPUS_PER_NODE),
                "NODE_RANK": str(self.node_rank),
                "NODE_WORLD_SIZE": str(NODE_COUNT),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "1"),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
                "TORCH_NCCL_BLOCKING_WAIT": "0",
                "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
            }
        )
        # The demo uses FileStore rather than TCPStore.  Do not inherit a
        # single-node MASTER_ADDR/MASTER_PORT from the caller.
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        env.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        if self.args.nccl_socket_ifname:
            env["NCCL_SOCKET_IFNAME"] = (
                self.args.nccl_socket_ifname
            )
        if self.args.nccl_ib_hca:
            env["NCCL_IB_HCA"] = self.args.nccl_ib_hca

        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--work-dir",
            str(self.work_dir),
            "--steps",
            str(self.args.steps),
            "--log-every",
            str(self.args.log_every),
            "--checkpoint-interval",
            str(self.args.checkpoint_interval),
            "--recovery-command",
            str(self.command_path),
            "--dist-timeout-seconds",
            str(self.args.dist_timeout_seconds),
            "--recovery-timeout-seconds",
            str(self.args.recovery_timeout_seconds),
        ]
        if replacement:
            command.append("--replacement")
        else:
            command.extend(
                [
                    "--fail-rank",
                    str(self.args.fail_rank),
                    "--fail-at-step",
                    str(self.args.fail_at_step),
                ]
            )
        return subprocess.Popen(
            command,
            cwd=str(TRAIN_SCRIPT.parent),
            env=env,
        )

    def _launch_initial_workers(self) -> None:
        self._write_state("starting")
        for rank in self.ranks:
            process = self._launch_worker(rank, replacement=False)
            self.processes[rank] = process
            self.initial_pids[rank] = process.pid
        self._write_state("running")
        self._log(
            "initial processes: "
            + ", ".join(
                f"rank {rank}/{_coordinates(rank)}=pid "
                f"{self.initial_pids[rank]}"
                for rank in self.ranks
            )
        )

    def _report_failure(
        self,
        rank: int,
        pid: int,
        returncode: int,
    ) -> None:
        failure_key = (rank, pid)
        if failure_key in self.reported_failures:
            return
        self.reported_failures.add(failure_key)
        status = read_json(
            self.work_dir / "status" / f"rank_{rank}.json"
        )
        is_replacement = (
            pid != self.initial_pids.get(rank)
        )
        record = {
            "rank": rank,
            "pid": pid,
            "old_pid": self.initial_pids.get(rank),
            "returncode": returncode,
            "node_rank": self.node_rank,
            "is_replacement": is_replacement,
            "status_phase": (
                status.get("phase") if status is not None else None
            ),
            "timestamp_unix_ms": int(time.time() * 1000),
        }
        atomic_write_json(
            self.failure_dir / f"rank_{rank}_{pid}.json",
            record,
        )
        self._log(
            f"FAILED rank {rank}/{_coordinates(rank)}: "
            f"pid={pid}, exit_code={returncode}"
        )

    def _poll_local_failures(self) -> None:
        for rank, process in self.processes.items():
            returncode = process.poll()
            if returncode is None:
                continue
            status = read_json(
                self.work_dir / "status" / f"rank_{rank}.json"
            )
            if (
                returncode == 0
                and status is not None
                and status.get("phase") == "finished"
            ):
                continue
            self._report_failure(
                rank,
                process.pid,
                returncode,
            )

    def _wait_for_peer_initial_pids(self) -> None:
        for peer in range(NODE_COUNT):
            state = _wait_for_json_matching(
                self.work_dir
                / "agents"
                / f"node_{peer}.json",
                self.args.job_timeout_seconds,
                f"node {peer} agent to start",
                lambda value, peer=peer: all(
                    str(rank) in value.get("initial_pids", {})
                    for rank in _node_ranks(peer)
                ),
            )
            if int(state.get("node_rank", -1)) != peer:
                raise RuntimeError(
                    f"node {peer} agent metadata is invalid"
                )
            initial_pids = state.get("initial_pids", {})
            for rank in _node_ranks(peer):
                if str(rank) not in initial_pids:
                    raise RuntimeError(
                        f"node {peer} did not report rank {rank}"
                    )
                self.peer_initial_pids[rank] = int(
                    initial_pids[str(rank)]
                )

    def _all_initial_pids(self) -> Dict[int, int]:
        result = dict(self.initial_pids)
        result.update(self.peer_initial_pids)
        if set(result) != set(range(WORLD_SIZE)):
            raise RuntimeError(
                "initial PID map does not cover the global world"
            )
        return result

    def _coordinate_recovery(
        self,
        failure: Dict[str, Any],
    ) -> None:
        if self.node_rank != 0:
            raise RuntimeError("only node 0 coordinates recovery")
        failed_rank = int(failure["rank"])
        if failed_rank != self.args.fail_rank:
            raise RuntimeError(
                f"unexpected failed rank {failed_rank}; "
                f"expected {self.args.fail_rank}"
            )
        if failure.get("is_replacement"):
            raise RuntimeError("the replacement rank failed")

        latest = _wait_for_json(
            self.work_dir / "checkpoints" / "latest.json",
            self.args.recovery_timeout_seconds,
            "a committed checkpoint",
        )
        now = time.time()
        command = RecoveryCommand(
            recovery_epoch=1,
            attempt_id=(
                f"two-node-{int(now * 1000)}-{failed_rank}"
            ),
            failed_rank=failed_rank,
            checkpoint_path=str(latest["checkpoint_path"]),
            checkpoint_step=int(latest["checkpoint_step"]),
            reconfigure_uuid=fresh_reconfigure_uuid(),
            world_size=WORLD_SIZE,
            deadline_unix_ms=int(
                (
                    now
                    + self.args.recovery_timeout_seconds
                )
                * 1000
            ),
        )
        self.recovery_started = True
        atomic_write_json(
            self.command_path,
            command.to_dict(),
        )
        self._write_state(
            "recovering",
            recovery_epoch=command.recovery_epoch,
            failed_rank=failed_rank,
        )
        initial_pids = self._all_initial_pids()
        healthy = {
            rank: pid
            for rank, pid in initial_pids.items()
            if rank != failed_rank
        }
        self._log(
            f"requested recovery epoch {command.recovery_epoch} "
            f"for rank {failed_rank}/{_coordinates(failed_rank)}"
        )
        self._log(
            "healthy ranks retain their initial processes: "
            + ", ".join(
                f"rank {rank}=pid {pid}"
                for rank, pid in sorted(healthy.items())
            )
        )

    def _unexpected_failure(
        self,
        failures: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        initial_pids = self._all_initial_pids()
        for failure in failures:
            rank = int(failure["rank"])
            pid = int(failure["pid"])
            is_expected_old_target = (
                rank == self.args.fail_rank
                and pid == initial_pids[rank]
                and not failure.get("is_replacement", False)
            )
            if not is_expected_old_target:
                return failure
        return None

    def _maybe_apply_recovery_command(self) -> None:
        try:
            command = load_recovery_command(self.command_path)
        except (TypeError, ValueError):
            return
        if (
            command is None
            or command.recovery_epoch <= self.last_command_epoch
        ):
            return
        if command.world_size != WORLD_SIZE:
            raise ValueError("recovery world size does not match")
        if command.failed_rank not in range(WORLD_SIZE):
            raise ValueError("recovery failure rank is invalid")

        self.last_command_epoch = command.recovery_epoch
        self.recovery_started = True
        self._write_state(
            "recovering",
            recovery_epoch=command.recovery_epoch,
            failed_rank=command.failed_rank,
        )
        self._log(
            f"observed recovery epoch {command.recovery_epoch}"
        )

        failed_rank = command.failed_rank
        if failed_rank not in self.ranks:
            self._log(
                f"rank {failed_rank} is owned by node "
                f"{failed_rank // GPUS_PER_NODE}; this node "
                "keeps all local processes running"
            )
            return

        old_process = self.processes.get(failed_rank)
        if old_process is None:
            raise RuntimeError(
                f"failed rank {failed_rank} is missing locally"
            )
        if old_process.poll() is None:
            raise RuntimeError(
                f"refusing to replace live rank {failed_rank}"
            )
        old_pid = old_process.pid
        replacement = self._launch_worker(
            failed_rank,
            replacement=True,
        )
        self.processes[failed_rank] = replacement
        self.replacement_pids[failed_rank] = replacement.pid
        self._write_state(
            "recovery_started",
            recovery_epoch=command.recovery_epoch,
            failed_rank=failed_rank,
        )
        self._log(
            f"REPLACED rank {failed_rank}/"
            f"{_coordinates(failed_rank)}: "
            f"old_pid={old_pid}, new_pid={replacement.pid}"
        )

    def _report_recoveries(self) -> None:
        for rank in self.ranks:
            if rank in self.reported_recoveries:
                continue
            status = read_json(
                self.work_dir / "status" / f"rank_{rank}.json"
            )
            if (
                status is not None
                and status.get("phase") == "running"
                and int(status.get("recovery_count", 0)) > 0
            ):
                self._log(
                    f"RECOVERED rank {rank}/"
                    f"{_coordinates(rank)}: pid={status['pid']}, "
                    f"epoch={status['recovery_epoch']}, "
                    f"step={status['global_step']}"
                )
                self.reported_recoveries.add(rank)

    def _local_processes_finished(self) -> bool:
        for rank, process in self.processes.items():
            if process.poll() is None:
                return False
            status = read_json(
                self.work_dir / "status" / f"rank_{rank}.json"
            )
            if status is None or status.get("phase") != "finished":
                return False
        return True

    def _wait_for_peer_finished(self) -> None:
        if self.node_rank != 0:
            return
        for peer in range(NODE_COUNT):
            _wait_for_json_matching(
                self.work_dir
                / "agents"
                / f"node_{peer}.json",
                self.args.job_timeout_seconds,
                f"node {peer} to finish",
                lambda value: value.get("phase") == "finished",
            )

    def _validate_global(self) -> None:
        initial_pids = self._all_initial_pids()
        states: Dict[int, Dict[str, Any]] = {}
        for node_rank in range(NODE_COUNT):
            state = read_json(
                self.work_dir
                / "agents"
                / f"node_{node_rank}.json"
            )
            if state is None:
                raise RuntimeError(
                    f"missing state for node {node_rank}"
                )
            states[node_rank] = state

        for rank in range(WORLD_SIZE):
            status = read_json(
                self.work_dir / "status" / f"rank_{rank}.json"
            )
            if status is None or status.get("phase") != "finished":
                raise RuntimeError(f"rank {rank} did not finish")
            if int(status.get("global_step", -1)) != self.args.steps:
                raise RuntimeError(
                    f"rank {rank} did not reach the final step"
                )
            if int(status.get("recovery_count", 0)) < 1:
                raise RuntimeError(
                    f"rank {rank} did not recover"
                )
            final_pid = int(status["pid"])
            if rank == self.args.fail_rank:
                if final_pid == initial_pids[rank]:
                    raise RuntimeError(
                        f"failed rank {rank} was not replaced"
                    )
                owner = rank // GPUS_PER_NODE
                owner_state = states[owner]
                replacement_pids = owner_state.get(
                    "replacement_pids",
                    {},
                )
                if (
                    str(rank) in replacement_pids
                    and int(replacement_pids[str(rank)])
                    != final_pid
                ):
                    raise RuntimeError(
                        f"rank {rank} replacement PID mismatch"
                    )
            elif final_pid != initial_pids[rank]:
                raise RuntimeError(
                    f"healthy rank {rank} changed PID"
                )

        print("")
        print("Two-node in-place restart summary")
        print("  topology: TP2 x PP2 x DP2, EP=1")
        print("  nodes: 2, GPUs per node: 4")
        print(f"  work_dir: {self.work_dir}")
        for rank in range(WORLD_SIZE):
            status = read_json(
                self.work_dir / "status" / f"rank_{rank}.json"
            )
            assert status is not None
            final_pid = int(status["pid"])
            initial_pid = initial_pids[rank]
            if final_pid == initial_pid:
                result = f"CONTINUED with pid {final_pid}"
            else:
                result = (
                    f"REPLACED {initial_pid} -> {final_pid}"
                )
            print(
                f"  rank {rank}/{_coordinates(rank)} "
                f"on node {rank // GPUS_PER_NODE}: {result}, "
                f"step={status.get('global_step')}, "
                f"recoveries={status.get('recovery_count')}"
            )

    def run(self) -> None:
        if self.node_rank == 0:
            self._initialize_run()
        else:
            self._wait_for_run()

        try:
            self._launch_initial_workers()
            self._wait_for_peer_initial_pids()
            deadline = (
                time.monotonic() + self.args.job_timeout_seconds
            )
            while True:
                self._poll_local_failures()
                failures = _read_all_failures(self.failure_dir)

                if (
                    self.node_rank == 0
                    and not self.recovery_started
                ):
                    target_failures = [
                        failure
                        for failure in failures
                        if int(failure["rank"])
                        == self.args.fail_rank
                        and not failure.get(
                            "is_replacement",
                            False,
                        )
                    ]
                    if target_failures:
                        self._coordinate_recovery(
                            target_failures[0]
                        )

                unexpected = self._unexpected_failure(failures)
                if unexpected is not None:
                    raise RuntimeError(
                        "unexpected rank failure: "
                        f"{unexpected}"
                    )

                self._maybe_apply_recovery_command()
                self._report_recoveries()

                if self._local_processes_finished():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "two-node in-place recovery demo timed out"
                    )
                time.sleep(0.2)

            exit_codes = {
                str(rank): process.returncode
                for rank, process in sorted(
                    self.processes.items()
                )
            }
            if any(code != 0 for code in exit_codes.values()):
                raise RuntimeError(
                    f"local workers exited unsuccessfully: "
                    f"{exit_codes}"
                )
            self._write_state(
                "finished",
                exit_codes=exit_codes,
            )
            self._log("local workers finished")
            self._wait_for_peer_finished()
            if self.node_rank == 0:
                self._validate_global()
                self._write_state("complete")
        finally:
            self._cleanup_workers()

    def _cleanup_workers(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-rank",
        type=int,
        required=True,
        choices=range(NODE_COUNT),
    )
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--fail-rank", type=int, default=5)
    parser.add_argument("--fail-at-step", type=int, default=23)
    parser.add_argument("--dist-timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--recovery-timeout-seconds",
        type=float,
        default=240.0,
    )
    parser.add_argument(
        "--job-timeout-seconds",
        type=float,
        default=1200.0,
    )
    parser.add_argument(
        "--nccl-socket-ifname",
        type=str,
        default="",
        help="optional NCCL_SOCKET_IFNAME for cross-node NCCL",
    )
    parser.add_argument(
        "--nccl-ib-hca",
        type=str,
        default="",
        help="optional NCCL_IB_HCA for InfiniBand",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fail_rank not in range(WORLD_SIZE):
        raise SystemExit("--fail-rank must be in [0, 7]")
    if args.fail_at_step < 0 or args.fail_at_step >= args.steps:
        raise SystemExit(
            "--fail-at-step must be nonnegative and less than --steps"
        )
    work_dir = Path(args.work_dir).resolve()
    agent = NodeAgent(args, work_dir)
    try:
        agent.run()
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        print(
            f"two-node demo failed on node {args.node_rank}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
