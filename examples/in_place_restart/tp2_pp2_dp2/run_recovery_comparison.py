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
"""Compare in-place recovery with restarting all eight training processes."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from recovery_protocol import atomic_write_json, read_json


MODES = ("in-place", "restart-all")


def _read_loss_trace(
    path: Path,
) -> Dict[int, Dict[str, Any]]:
    """Return the last recorded value for each training step."""

    values: Dict[int, Dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                values[int(value["global_step"])] = value
    except FileNotFoundError as error:
        raise RuntimeError(f"loss trace is missing: {path}") from error
    return values


def _compare_losses(
    in_place_path: Path,
    restart_all_path: Path,
    checkpoint_step: int,
    tolerance: float,
) -> Dict[str, Any]:
    in_place = _read_loss_trace(in_place_path)
    restart_all = _read_loss_trace(restart_all_path)
    common_steps = sorted(set(in_place) & set(restart_all))
    post_checkpoint_steps = [
        step for step in common_steps if step > checkpoint_step
    ]

    def _statistics(steps: List[int]) -> Dict[str, Any]:
        if not steps:
            return {
                "steps": [],
                "count": 0,
                "max_abs_diff": None,
                "max_rel_diff": None,
                "within_tolerance": False,
            }
        differences = []
        relative = []
        detail = []
        for step in steps:
            left = float(in_place[step]["loss"])
            right = float(restart_all[step]["loss"])
            absolute = abs(left - right)
            scale = max(abs(left), abs(right), 1e-12)
            differences.append(absolute)
            relative.append(absolute / scale)
            detail.append(
                {
                    "step": step,
                    "in_place_loss": left,
                    "restart_all_loss": right,
                    "abs_diff": absolute,
                    "rel_diff": absolute / scale,
                }
            )
        return {
            "steps": steps,
            "count": len(steps),
            "max_abs_diff": max(differences),
            "max_rel_diff": max(relative),
            "within_tolerance": max(differences) <= tolerance,
            "details": detail,
        }

    return {
        "common_steps": common_steps,
        "post_checkpoint": _statistics(post_checkpoint_steps),
        "all_common_steps": _statistics(common_steps),
        "tolerance": tolerance,
    }


def _run_mode(
    root: Path,
    mode: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    mode_dir = root / mode.replace("-", "_")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_in_place_demo.py")),
        "--work-dir",
        str(mode_dir),
        "--recovery-mode",
        mode,
        "--steps",
        str(args.steps),
        "--log-every",
        str(args.log_every),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--fail-rank",
        str(args.fail_rank),
        "--fail-at-step",
        str(args.fail_at_step),
        "--dist-timeout-seconds",
        str(args.dist_timeout_seconds),
        "--recovery-timeout-seconds",
        str(args.recovery_timeout_seconds),
        "--job-timeout-seconds",
        str(args.job_timeout_seconds),
    ]
    print("")
    print(f"=== Running recovery mode: {mode} ===")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"recovery mode {mode} failed with "
            f"exit code {completed.returncode}"
        )
    summary = read_json(mode_dir / "metrics" / "summary.json")
    if summary is None:
        raise RuntimeError(
            f"recovery mode {mode} did not write a summary"
        )
    return summary


def _recovery_delta(
    in_place: Dict[str, Any],
    restart_all: Dict[str, Any],
) -> Dict[str, Any]:
    in_place_seconds = in_place.get(
        "failure_to_all_resumed_seconds"
    )
    restart_all_seconds = restart_all.get(
        "failure_to_all_resumed_seconds"
    )
    saved = None
    if (
        in_place_seconds is not None
        and restart_all_seconds is not None
    ):
        saved = restart_all_seconds - in_place_seconds
    return {
        "in_place_failure_to_all_resumed_seconds": in_place_seconds,
        "restart_all_failure_to_all_resumed_seconds": restart_all_seconds,
        "failure_to_all_resumed_seconds_saved": saved,
        "in_place_process_restarts": in_place.get(
            "process_restarts"
        ),
        "restart_all_process_restarts": restart_all.get(
            "process_restarts"
        ),
        "process_restarts_avoided": (
            int(restart_all.get("process_restarts", 0))
            - int(in_place.get("process_restarts", 0))
        ),
        "in_place_total_wall_seconds": in_place.get(
            "total_wall_seconds"
        ),
        "restart_all_total_wall_seconds": restart_all.get(
            "total_wall_seconds"
        ),
    }


def _print_result(result: Dict[str, Any]) -> None:
    loss = result["loss"]["post_checkpoint"]
    recovery = result["recovery"]

    def _seconds(value: Any) -> str:
        return f"{value:.3f}s" if value is not None else "n/a"

    print("")
    print("Recovery comparison")
    print(f"  checkpoint step: {result['checkpoint_step']}")
    print(
        f"  loss steps compared: {loss['count']} "
        f"({loss['steps'][0]}..{loss['steps'][-1]})"
        if loss["steps"]
        else "  loss steps compared: 0"
    )
    print(
        "  max loss abs diff: "
        f"{loss['max_abs_diff']}"
    )
    print(
        "  max loss rel diff: "
        f"{loss['max_rel_diff']}"
    )
    print(
        "  loss within tolerance: "
        f"{loss['within_tolerance']}"
    )
    print(
        "  in-place failure-to-resume: "
        f"{_seconds(recovery['in_place_failure_to_all_resumed_seconds'])}"
    )
    print(
        "  restart-all failure-to-resume: "
        f"{_seconds(recovery['restart_all_failure_to_all_resumed_seconds'])}"
    )
    print(
        "  failure-to-resume saved: "
        f"{_seconds(recovery['failure_to_all_resumed_seconds_saved'])}"
    )
    print(
        "  process restarts avoided: "
        f"{recovery['process_restarts_avoided']}"
    )
    print(
        "  in-place total wall time: "
        f"{_seconds(recovery['in_place_total_wall_seconds'])}"
    )
    print(
        "  restart-all total wall time: "
        f"{_seconds(recovery['restart_all_total_wall_seconds'])}"
    )


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
    parser.add_argument(
        "--loss-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--run-order",
        choices=("in-place-first", "restart-all-first"),
        default="in-place-first",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fail_rank < 0 or args.fail_rank >= 8:
        raise SystemExit("--fail-rank must be in [0, 7]")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.checkpoint_interval <= 0:
        raise SystemExit("--checkpoint-interval must be positive")
    if args.fail_at_step < 0 or args.fail_at_step >= args.steps:
        raise SystemExit(
            "--fail-at-step must be nonnegative and less than --steps"
        )

    if args.work_dir:
        root = Path(args.work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise SystemExit(
                f"comparison work directory is not empty: {root}"
            )
    else:
        root = Path(
            tempfile.mkdtemp(prefix="dlrover-recovery-compare-")
        ).resolve()

    try:
        if args.run_order == "in-place-first":
            in_place = _run_mode(root, "in-place", args)
            restart_all = _run_mode(root, "restart-all", args)
        else:
            restart_all = _run_mode(root, "restart-all", args)
            in_place = _run_mode(root, "in-place", args)
        checkpoint_step = in_place.get("recovery_checkpoint_step")
        if checkpoint_step is None:
            raise RuntimeError(
                "in-place summary does not record the recovery checkpoint"
            )
        checkpoint_step = int(checkpoint_step)
        if int(
            restart_all.get("recovery_checkpoint_step", -1)
        ) != checkpoint_step:
            raise RuntimeError(
                "the two modes resumed from different checkpoints"
            )
        loss = _compare_losses(
            root / "in_place" / "metrics" / "loss_rank_2.jsonl",
            root / "restart_all" / "metrics" / "loss_rank_2.jsonl",
            checkpoint_step,
            args.loss_tolerance,
        )
        result = {
            "work_dir": str(root),
            "checkpoint_step": checkpoint_step,
            "in_place_summary": in_place,
            "restart_all_summary": restart_all,
            "loss": loss,
            "recovery": _recovery_delta(
                in_place,
                restart_all,
            ),
        }
        atomic_write_json(root / "comparison.json", result)
        _print_result(result)
        print(f"  comparison file: {root / 'comparison.json'}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"recovery comparison failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
