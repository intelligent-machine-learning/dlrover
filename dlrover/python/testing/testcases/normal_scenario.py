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

"""Normal training scenario — no failures.

All workers join the master, send heartbeats for the configured duration,
then report success and exit cleanly.

Run with local master (default):

    python -m dlrover.python.testing.testcases.normal_scenario

Run with DistributedJobMaster:

    python -m dlrover.python.testing.testcases.normal_scenario --dist

Optional flags:

    --num_workers   N     Number of simulated workers  (default: 4)
    --run_duration  SEC   How long each worker runs     (default: 5.0)
"""

import argparse

from dlrover.python.common.log import default_logger as logger
from dlrover.python.testing.agent.test_agent import (
    AgentOutcome,
    make_agents,
    run_agents,
)
from dlrover.python.testing.master.master_setup import MasterContext


def scenario_normal_training(
    num_workers: int = 4,
    run_duration: float = 5.0,
    master_type: str = "local",
):
    """Start master + N workers; all workers succeed normally.

    Args:
        num_workers: Number of simulated worker agents.
        run_duration: Seconds each agent runs before reporting success.
        master_type: ``"local"`` or ``"dist"``.

    Returns:
        List of ``AgentResult`` objects, one per agent.
    """
    logger.info(
        f"=== normal_training: {num_workers} workers, "
        f"run_duration={run_duration}s, master={master_type} ==="
    )

    with MasterContext(master_type=master_type, num_workers=num_workers) as ctx:
        agents = make_agents(
            ctx.addr,
            num_agents=num_workers,
            run_duration=run_duration,
        )
        results = run_agents(agents)

    succeeded = sum(1 for r in results if r.outcome == AgentOutcome.SUCCEEDED)
    failed = sum(1 for r in results if r.outcome == AgentOutcome.FAILED)
    logger.info(f"  succeeded={succeeded}  failed={failed}  total={num_workers}")
    for r in results:
        logger.info(
            f"  agent-{r.agent_id}: {r.outcome.name}  "
            f"heartbeats={r.heartbeat_count}  elapsed={r.elapsed:.1f}s"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal training scenario")
    parser.add_argument(
        "--dist",
        action="store_true",
        help="Use DistributedJobMaster instead of LocalJobMaster",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_duration", type=float, default=5.0)
    args = parser.parse_args()

    scenario_normal_training(
        num_workers=args.num_workers,
        run_duration=args.run_duration,
        master_type="dist" if args.dist else "local",
    )
