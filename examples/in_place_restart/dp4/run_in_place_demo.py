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
"""Launch a local four-rank in-place recovery demonstration.

The supervisor models the DLRover control-plane responsibilities needed by
this example: it preserves healthy workers, restarts only the failed logical
rank, publishes a committed checkpoint, and supplies a fresh recovery epoch
and reconfigure UUID.
"""

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


WORLD_SIZE = 4


def _wait_for_file(path: Path, timeout_seconds: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = read_json(path)
        if value is not None:
            return value
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
            # nccl2 owns failure handling in reconfigure mode. Leaving legacy
            # process-abort handling enabled would obscure the demonstration.
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
            "TORCH_NCCL_BLOCKING_WAIT": "0",
        }
    )
    script_dir = Path(__file__).resolve().parent
    worker_script = script_dir / "moe_train.py"
    command = [
        sys.executable,
        str(worker_script),
        "--work-dir",
        str(work_dir),
        "--steps",
        str(args.steps),
        "--log-every",
        str(args.log_every),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--backend",
        "nccl2",
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


def _print_summary(
    work_dir: Path,
    initial_pids: Dict[int, int],
    final_pids: Dict[int, int],
    recovery_started: bool,
) -> None:
    print("")
    print("In-place restart demo summary")
    print(f"  work_dir: {work_dir}")
    print(f"  recovery_started: {recovery_started}")
    for rank in range(WORLD_SIZE):
        status = read_json(work_dir / "status" / f"rank_{rank}.json")
        status = status or {}
        if initial_pids[rank] != final_pids[rank]:
            print(
                f"  rank {rank}: REPLACED "
                f"{initial_pids[rank]} -> {final_pids[rank]}, "
                f"step={status.get('global_step')}, "
                f"recoveries={status.get('recovery_count')}"
            )
        else:
            print(
                f"  rank {rank}: CONTINUED with pid {final_pids[rank]}, "
                f"step={status.get('global_step')}, "
                f"recoveries={status.get('recovery_count')}"
            )


def run(args: argparse.Namespace) -> int:
    if args.fail_rank < 0:
        raise ValueError("this demo requires failure injection")
    if args.fail_at_step >= args.steps:
        raise ValueError("--fail-at-step must be less than --steps")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")

    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        if any(work_dir.iterdir()):
            raise ValueError(
                f"work directory is not empty: {work_dir}; "
                "use a fresh directory"
            )
    else:
        work_dir = Path(
            tempfile.mkdtemp(prefix="dlrover-in-place-moe-")
        ).resolve()

    recovery_dir = work_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    recovery_command_path = recovery_dir / "command.json"

    processes: Dict[int, subprocess.Popen] = {}
    initial_pids: Dict[int, int] = {}
    final_pids: Dict[int, int] = {}
    try:
        for rank in range(WORLD_SIZE):
            process = _launch_worker(
                rank,
                args,
                work_dir,
                recovery_command_path,
                replacement=False,
            )
            processes[rank] = process
            initial_pids[rank] = process.pid
        print(
            "[supervisor] initial processes: "
            + ", ".join(
                f"rank {rank}=pid {initial_pids[rank]}"
                for rank in range(WORLD_SIZE)
            )
        )

        deadline = time.monotonic() + args.job_timeout_seconds
        recovery_started = False
        replacement_pid = None
        reported_recovery = set()

        while True:
            alive = [
                rank
                for rank, process in processes.items()
                if process.poll() is None
            ]
            if not alive:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("in-place recovery demo timed out")

            failed_processes = {
                rank: process
                for rank, process in processes.items()
                if process.poll() is not None
            }
            for rank, process in failed_processes.items():
                return_code = process.returncode
                if (
                    rank == args.fail_rank
                    and not recovery_started
                    and return_code != 0
                ):
                    healthy_pids = {
                        healthy_rank: healthy_process.pid
                        for healthy_rank, healthy_process in (
                            processes.items()
                        )
                        if healthy_rank != rank
                        and healthy_process.poll() is None
                    }
                    latest = _wait_for_file(
                        work_dir / "checkpoints" / "latest.json",
                        timeout_seconds=args.recovery_timeout_seconds,
                    )
                    command = RecoveryCommand(
                        recovery_epoch=1,
                        attempt_id=(
                            f"attempt-{int(time.time() * 1000)}-{rank}"
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
                        recovery_command_path,
                        command.to_dict(),
                    )
                    print(
                        f"[supervisor] FAILED rank {rank}: "
                        f"old_pid={process.pid}, exit_code={return_code}"
                    )
                    print(
                        "[supervisor] healthy processes still running: "
                        + ", ".join(
                            f"rank {healthy_rank}=pid {pid}"
                            for healthy_rank, pid in sorted(
                                healthy_pids.items()
                            )
                        )
                    )
                    replacement = _launch_worker(
                        rank,
                        args,
                        work_dir,
                        recovery_command_path,
                        replacement=True,
                    )
                    processes[rank] = replacement
                    replacement_pid = replacement.pid
                    recovery_started = True
                    print(
                        f"[supervisor] REPLACED rank {rank}: "
                        f"old_pid={process.pid}, "
                        f"new_pid={replacement.pid}"
                    )
                elif not recovery_started and return_code != 0:
                    raise RuntimeError(
                        f"rank {rank} failed before recovery with "
                        f"exit code {return_code}"
                    )
                elif (
                    recovery_started
                    and rank != args.fail_rank
                    and return_code != 0
                ):
                    raise RuntimeError(
                        f"healthy rank {rank} failed during recovery with "
                        f"exit code {return_code}"
                    )
                elif (
                    recovery_started
                    and rank == args.fail_rank
                    and return_code != 0
                ):
                    raise RuntimeError(
                        f"replacement rank {rank} failed during recovery "
                        f"with exit code {return_code}"
                    )

            for rank in range(WORLD_SIZE):
                if rank in reported_recovery:
                    continue
                status = read_json(
                    work_dir / "status" / f"rank_{rank}.json"
                )
                if (
                    status is not None
                    and int(status.get("recovery_count", 0)) > 0
                    and status.get("phase") == "running"
                ):
                    print(
                        f"[supervisor] RECOVERED rank {rank}: "
                        f"pid={status['pid']}, "
                        f"epoch={status['recovery_epoch']}, "
                        f"step={status['global_step']}"
                    )
                    reported_recovery.add(rank)

            time.sleep(0.2)

        for rank, process in processes.items():
            final_pids[rank] = process.pid
            if process.returncode != 0:
                raise RuntimeError(
                    f"rank {rank} exited with code {process.returncode}"
                )

        for rank in range(WORLD_SIZE):
            status = read_json(
                work_dir / "status" / f"rank_{rank}.json"
            )
            if status is None:
                raise RuntimeError(f"rank {rank} did not publish status")
            if status.get("phase") != "finished":
                raise RuntimeError(
                    f"rank {rank} did not finish: {status}"
                )
            if int(status.get("recovery_count", 0)) < 1:
                raise RuntimeError(
                    f"rank {rank} did not complete recovery"
                )
            if rank != args.fail_rank:
                if int(status["pid"]) != initial_pids[rank]:
                    raise RuntimeError(
                        f"healthy rank {rank} changed PID"
                    )
            elif replacement_pid is None or int(status["pid"]) != (
                replacement_pid
            ):
                raise RuntimeError(
                    f"failed rank {rank} was not replaced correctly"
                )

        _print_summary(
            work_dir,
            initial_pids,
            final_pids,
            recovery_started,
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
    parser.add_argument("--fail-rank", type=int, default=2)
    parser.add_argument("--fail-at-step", type=int, default=23)
    parser.add_argument("--dist-timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--recovery-timeout-seconds", type=float, default=180.0
    )
    parser.add_argument(
        "--job-timeout-seconds", type=float, default=900.0
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fail_rank < 0 or args.fail_rank >= WORLD_SIZE:
        raise SystemExit("--fail-rank must be between 0 and 3")
    try:
        raise SystemExit(run(args))
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"in-place restart demo failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
