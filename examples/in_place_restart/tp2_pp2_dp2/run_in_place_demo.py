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
"""Run the TP2 x PP2 x DP2 in-place recovery demonstration."""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from recovery_protocol import (
    RecoveryCommand,
    atomic_write_json,
    fresh_reconfigure_uuid,
    read_json,
)


WORLD_SIZE = 8
TP_SIZE = 2
PP_SIZE = 2
DP_SIZE = 2


def _coordinates(rank: int) -> str:
    tp_rank = rank % TP_SIZE
    pp_rank = (rank // TP_SIZE) % PP_SIZE
    dp_rank = rank // (TP_SIZE * PP_SIZE)
    return f"TP{tp_rank}-PP{pp_rank}-DP{dp_rank}"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_file(path: Path, timeout_seconds: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = read_json(path)
        if value is not None:
            return value
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def _launch_worker(
    rank: int,
    args: argparse.Namespace,
    work_dir: Path,
    recovery_command_path: Path,
    replacement: bool,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_find_free_port()),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "1"),
            "NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_BLOCKING_WAIT": "0",
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
        }
    )
    script_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(script_dir / "hybrid_moe_train.py"),
        "--work-dir",
        str(work_dir),
        "--steps",
        str(args.steps),
        "--log-every",
        str(args.log_every),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--recovery-command",
        str(recovery_command_path),
        "--dist-timeout-seconds",
        str(args.dist_timeout_seconds),
        "--recovery-timeout-seconds",
        str(args.recovery_timeout_seconds),
    ]
    if replacement:
        command.append("--replacement")
    else:
        command.extend(
            [
                "--fail-rank",
                str(args.fail_rank),
                "--fail-at-step",
                str(args.fail_at_step),
            ]
        )
    return subprocess.Popen(
        command,
        cwd=str(script_dir),
        env=env,
    )


def _cleanup_workers(
    processes: Dict[int, subprocess.Popen],
) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(args: argparse.Namespace) -> int:
    if args.fail_rank < 0 or args.fail_rank >= WORLD_SIZE:
        raise ValueError("invalid failure rank")
    if args.fail_at_step >= args.steps:
        raise ValueError("fail step must be less than total steps")

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        if any(work_dir.iterdir()):
            raise ValueError(
                f"work directory is not empty: {work_dir}"
            )
    else:
        work_dir = Path(
            tempfile.mkdtemp(prefix="dlrover-hybrid-moe-")
        ).resolve()

    recovery_dir = work_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    command_path = recovery_dir / "command.json"

    processes: Dict[int, subprocess.Popen] = {}
    initial_pids: Dict[int, int] = {}
    replacement_pid = None
    recovery_started = False
    reported_recovery = set()

    try:
        for rank in range(WORLD_SIZE):
            process = _launch_worker(
                rank,
                args,
                work_dir,
                command_path,
                replacement=False,
            )
            processes[rank] = process
            initial_pids[rank] = process.pid

        print(
            "[supervisor] topology: TP2 x PP2 x DP2, EP=1",
            flush=True,
        )
        print(
            "[supervisor] initial processes: "
            + ", ".join(
                f"rank {rank}/{_coordinates(rank)}"
                f"=pid {initial_pids[rank]}"
                for rank in range(WORLD_SIZE)
            ),
            flush=True,
        )

        deadline = time.monotonic() + args.job_timeout_seconds
        while True:
            alive = [
                rank
                for rank, process in processes.items()
                if process.poll() is None
            ]
            if not alive:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("hybrid recovery demo timed out")

            failed = {
                rank: process
                for rank, process in processes.items()
                if process.poll() is not None
            }
            for rank, process in failed.items():
                return_code = process.returncode
                if (
                    rank == args.fail_rank
                    and not recovery_started
                    and return_code != 0
                ):
                    healthy = {
                        peer: peer_process.pid
                        for peer, peer_process in processes.items()
                        if peer != rank
                        and peer_process.poll() is None
                    }
                    latest = _wait_for_file(
                        work_dir / "checkpoints" / "latest.json",
                        args.recovery_timeout_seconds,
                    )
                    command = RecoveryCommand(
                        recovery_epoch=1,
                        attempt_id=(
                            f"hybrid-{int(time.time() * 1000)}-{rank}"
                        ),
                        failed_rank=rank,
                        checkpoint_path=latest["checkpoint_path"],
                        checkpoint_step=int(latest["checkpoint_step"]),
                        reconfigure_uuid=fresh_reconfigure_uuid(),
                        world_size=WORLD_SIZE,
                        deadline_unix_ms=int(
                            (
                                time.time()
                                + args.recovery_timeout_seconds
                            )
                            * 1000
                        ),
                    )
                    atomic_write_json(
                        command_path,
                        command.to_dict(),
                    )
                    print(
                        f"[supervisor] FAILED rank {rank}/"
                        f"{_coordinates(rank)}: old_pid={process.pid}, "
                        f"exit_code={return_code}",
                        flush=True,
                    )
                    print(
                        "[supervisor] healthy processes still running: "
                        + ", ".join(
                            f"rank {peer}=pid {pid}"
                            for peer, pid in sorted(healthy.items())
                        ),
                        flush=True,
                    )
                    replacement = _launch_worker(
                        rank,
                        args,
                        work_dir,
                        command_path,
                        replacement=True,
                    )
                    processes[rank] = replacement
                    replacement_pid = replacement.pid
                    recovery_started = True
                    print(
                        f"[supervisor] REPLACED rank {rank}/"
                        f"{_coordinates(rank)}: old_pid={process.pid}, "
                        f"new_pid={replacement.pid}",
                        flush=True,
                    )
                elif not recovery_started and return_code != 0:
                    raise RuntimeError(
                        f"rank {rank} failed before recovery with "
                        f"exit code {return_code}"
                    )
                elif (
                    recovery_started
                    and return_code != 0
                    and rank != args.fail_rank
                ):
                    raise RuntimeError(
                        f"healthy rank {rank} failed during recovery"
                    )
                elif (
                    recovery_started
                    and return_code != 0
                    and rank == args.fail_rank
                ):
                    raise RuntimeError(
                        f"replacement rank {rank} failed"
                    )

            for rank in range(WORLD_SIZE):
                if rank in reported_recovery:
                    continue
                status = read_json(
                    work_dir / "status" / f"rank_{rank}.json"
                )
                if (
                    status is not None
                    and status.get("phase") == "running"
                    and int(status.get("recovery_count", 0)) > 0
                ):
                    print(
                        f"[supervisor] RECOVERED rank {rank}/"
                        f"{_coordinates(rank)}: pid={status['pid']}, "
                        f"epoch={status['recovery_epoch']}, "
                        f"step={status['global_step']}",
                        flush=True,
                    )
                    reported_recovery.add(rank)

            time.sleep(0.2)

        for rank, process in processes.items():
            if process.returncode != 0:
                raise RuntimeError(
                    f"rank {rank} exited with {process.returncode}"
                )
            status = read_json(
                work_dir / "status" / f"rank_{rank}.json"
            )
            if status is None or status.get("phase") != "finished":
                raise RuntimeError(f"rank {rank} did not finish")
            if int(status.get("recovery_count", 0)) < 1:
                raise RuntimeError(f"rank {rank} did not recover")
            if rank != args.fail_rank:
                if int(status["pid"]) != initial_pids[rank]:
                    raise RuntimeError(
                        f"healthy rank {rank} changed PID"
                    )
            elif replacement_pid is None or int(
                status["pid"]
            ) != replacement_pid:
                raise RuntimeError(
                    f"rank {rank} replacement PID is incorrect"
                )

        print("")
        print("Hybrid in-place restart summary")
        print(f"  topology: TP2 x PP2 x DP2, EP=1")
        print(f"  work_dir: {work_dir}")
        for rank in range(WORLD_SIZE):
            status = read_json(
                work_dir / "status" / f"rank_{rank}.json"
            )
            status = status or {}
            final_pid = int(status["pid"])
            if final_pid == initial_pids[rank]:
                result = f"CONTINUED with pid {final_pid}"
            else:
                result = (
                    f"REPLACED {initial_pids[rank]} -> {final_pid}"
                )
            print(
                f"  rank {rank}/{_coordinates(rank)}: {result}, "
                f"step={status.get('global_step')}, "
                f"recoveries={status.get('recovery_count')}"
            )
        return 0
    finally:
        _cleanup_workers(processes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, default="")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--fail-rank", type=int, default=3)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        print(f"hybrid demo failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
