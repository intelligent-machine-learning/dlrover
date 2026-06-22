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

"""Example scenario: many workers fail simultaneously with large error dumps.

This is the scenario that causes master OOM in production:
  - N workers all fail within a short window
  - Each failure carries a large ``error_data`` payload (dump string)
  - The master stores the payload inside the Node object via set_exit_reason()
  - Those Node objects are deep-copied into _job_nodes and never evicted

Run directly to observe master memory growth:

    python -m dlrover.python.testing.testcases.example_oom_scenario

Pass ``--dist`` to exercise DistributedJobMaster instead of LocalJobMaster:

    python -m dlrover.python.testing.testcases.example_oom_scenario --dist

Control payload size (default 50 MB per worker):

    python -m dlrover.python.testing.testcases.example_oom_scenario --payload_mb 50

Typical observation before the fix:
    master RSS after N failures ≈ N × payload_bytes
"""

import argparse
import time

from dlrover.python.common.log import default_logger as logger
from dlrover.python.testing.agent.test_agent import (
    AgentOutcome,
    FailureSpec,
    make_agents,
    make_mixed_agents,
    run_agents,
)
from dlrover.python.testing.master.master_setup import MasterContext

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _master_rss_mb(ctx: MasterContext) -> float:
    """Return master subprocess RSS in MB, or -1 if psutil is unavailable."""
    if not _PSUTIL_AVAILABLE or ctx.master_proc is None:
        return -1.0
    try:
        return (
            psutil.Process(ctx.master_proc.pid).memory_info().rss / 1024 / 1024
        )
    except psutil.NoSuchProcess:
        return -1.0


def scenario_all_succeed(
    num_workers: int = 4,
    run_duration: float = 3.0,
    master_type: str = "local",
):
    """Baseline: all workers succeed normally."""
    logger.info(
        f"=== scenario_all_succeed: {num_workers} workers "
        f"(master={master_type}) ==="
    )
    with MasterContext(
        master_type=master_type, num_workers=num_workers
    ) as ctx:
        agents = make_agents(
            ctx.addr,
            num_agents=num_workers,
            run_duration=run_duration,
        )
        results = run_agents(agents)

    succeeded = sum(1 for r in results if r.outcome == AgentOutcome.SUCCEEDED)
    logger.info(f"  succeeded={succeeded}/{num_workers}")
    return results


def scenario_burst_failures(
    num_workers: int = 8,
    payload_bytes: int = 50 * 1024 * 1024,  # 50 MB per worker
    fail_after_heartbeats: int = 1,
    run_duration: float = 5.0,
    master_type: str = "local",
):
    """All workers fail simultaneously with a large dump payload.

    This reproduces the OOM bug:
      - Each failure stores ``payload_bytes`` inside node.exit_reason
      - Nodes accumulate in _job_nodes (never evicted)
      - Total memory pinned ≈ num_workers × payload_bytes

    Master subprocess RSS is sampled via psutil (if installed) so the
    measurement reflects the actual process that accumulates the memory,
    not the parent test process.
    """
    payload = "X" * payload_bytes
    spec = FailureSpec(
        error_data=payload,
        after_heartbeats=fail_after_heartbeats,
    )

    logger.info(
        f"=== scenario_burst_failures: {num_workers} workers, "
        f"{payload_bytes // 1024 // 1024} MB payload each "
        f"(expected ~{num_workers * payload_bytes // 1024 // 1024} MB total) "
        f"(master={master_type}) ==="
    )

    with MasterContext(
        master_type=master_type, num_workers=num_workers
    ) as ctx:
        rss_before = _master_rss_mb(ctx)
        if rss_before >= 0:
            logger.info(f"  master RSS before failures: {rss_before:.1f} MB")

        agents = make_agents(
            ctx.addr,
            num_agents=num_workers,
            failure_spec=spec,
            run_duration=run_duration,
        )
        results = run_agents(agents)

        # Let the master process the failures before we sample RSS.
        time.sleep(1.0)

        rss_after = _master_rss_mb(ctx)

    failed = sum(1 for r in results if r.outcome == AgentOutcome.FAILED)
    logger.info(f"  failed={failed}/{num_workers}")

    if rss_after >= 0:
        increase = rss_after - rss_before
        logger.info(
            f"  master RSS: {rss_before:.1f} MB → {rss_after:.1f} MB "
            f"(+{increase:.1f} MB, "
            f"expected ~{num_workers * payload_bytes / 1024 / 1024:.0f} MB)"
        )
    else:
        logger.info(
            "  (install psutil to measure master subprocess RSS: "
            "pip install psutil)"
        )

    return results, rss_after - rss_before if rss_after >= 0 else 0.0


def scenario_partial_failures(
    num_workers: int = 8,
    num_failing: int = 4,
    payload_bytes: int = 50 * 1024 * 1024,  # 50 MB per worker
    run_duration: float = 5.0,
    master_type: str = "local",
):
    """A subset of workers fail while the rest succeed."""
    payload = "Y" * payload_bytes
    spec = FailureSpec(error_data=payload, after_heartbeats=1)

    fail_indices = list(range(num_failing))

    logger.info(
        f"=== scenario_partial_failures: {num_failing}/{num_workers} fail, "
        f"{payload_bytes // 1024 // 1024} MB payload "
        f"(master={master_type}) ==="
    )
    with MasterContext(
        master_type=master_type, num_workers=num_workers
    ) as ctx:
        agents = make_mixed_agents(
            ctx.addr,
            num_agents=num_workers,
            fail_indices=fail_indices,
            failure_spec=spec,
            run_duration=run_duration,
        )
        results = run_agents(agents)

    failed = sum(1 for r in results if r.outcome == AgentOutcome.FAILED)
    succeeded = sum(1 for r in results if r.outcome == AgentOutcome.SUCCEEDED)
    logger.info(f"  succeeded={succeeded}, failed={failed}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist",
        action="store_true",
        help="Use DistributedJobMaster instead of LocalJobMaster",
    )
    parser.add_argument(
        "--payload_mb",
        type=int,
        default=50,
        help="Error payload size per worker in MB (default: 50)",
    )
    args = parser.parse_args()
    master_type = "dist" if args.dist else "local"
    payload_bytes = args.payload_mb * 1024 * 1024

    scenario_all_succeed(num_workers=4, master_type=master_type)

    print()
    _, mem_increase = scenario_burst_failures(
        num_workers=8,
        payload_bytes=payload_bytes,
        master_type=master_type,
    )

    print()
    scenario_partial_failures(
        num_workers=8,
        num_failing=4,
        payload_bytes=payload_bytes,
        master_type=master_type,
    )
