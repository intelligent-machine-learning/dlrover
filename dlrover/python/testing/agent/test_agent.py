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

"""Simulated training agents for use in integration tests.

Each TestTrainingAgent runs on its own thread and communicates with the master
via gRPC, mimicking the heartbeat / rendezvous / failure-reporting behaviour
of a real ElasticTrainingAgent without spawning actual GPU processes.
"""

import time
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from dlrover.python.common.constants import (
    NodeType,
    RendezvousName,
    TrainingExceptionLevel,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import GrpcMasterClient


class AgentOutcome(Enum):
    PENDING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass
class AgentResult:
    agent_id: int
    outcome: AgentOutcome
    error: Optional[str] = None
    heartbeat_count: int = 0
    elapsed: float = 0.0


@dataclass
class FailureSpec:
    """Describes how and when an agent should report a failure.

    Attributes:
        error_data: The payload sent as ``error_data`` to the master.
            Typically a JSON-serialised stack trace or dump string.
        level: ``TrainingExceptionLevel`` constant.
        after_heartbeats: Fail after this many successful heartbeats.
            Defaults to 0 (fail immediately on the first beat).
    """

    error_data: str = "simulated process error"
    level: str = TrainingExceptionLevel.PROCESS_ERROR
    after_heartbeats: int = 0


class TestTrainingAgent:
    """Simulates one worker node interacting with a DLRover master.

    The agent lifecycle on each run:
    1. Join the training rendezvous.
    2. Send periodic heartbeats (every ``heartbeat_interval`` seconds).
    3. After ``failure_spec.after_heartbeats`` heartbeats: report the failure
       and exit, OR (if no failure_spec) keep sending heartbeats until
       ``run_duration`` seconds have elapsed, then report success.

    Create a fleet of agents and call ``run_all()`` to drive them all
    concurrently on daemon threads.
    """

    def __init__(
        self,
        agent_id: int,
        master_addr: str,
        num_agents: int,
        failure_spec: Optional[FailureSpec] = None,
        heartbeat_interval: float = 1.0,
        run_duration: float = 5.0,
        rdzv_name: str = RendezvousName.TRAINING,
    ):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.failure_spec = failure_spec
        self.heartbeat_interval = heartbeat_interval
        self.run_duration = run_duration
        self.rdzv_name = rdzv_name

        self.result = AgentResult(
            agent_id=agent_id, outcome=AgentOutcome.PENDING
        )

        # Each agent creates its own gRPC client so agents can run in the same
        # process without singleton conflicts.
        self._client = GrpcMasterClient(
            master_addr,
            node_id=agent_id,
            node_type=NodeType.WORKER,
        )

    def run(self) -> AgentResult:
        """Execute the full agent lifecycle synchronously."""
        start = time.monotonic()
        try:
            self._join_rendezvous()
            self._run_loop()
        except Exception as exc:
            logger.warning(f"[agent-{self.agent_id}] unhandled error: {exc}")
            self.result.outcome = AgentOutcome.FAILED
            self.result.error = str(exc)
        finally:
            self.result.elapsed = time.monotonic() - start
        return self.result

    # ------------------------------------------------------------------
    # Lifecycle steps
    # ------------------------------------------------------------------

    def _join_rendezvous(self) -> None:
        try:
            self._client.join_rendezvous(
                node_rank=self.agent_id,
                local_world_size=1,
                rdzv_name=self.rdzv_name,
            )
            logger.debug(f"[agent-{self.agent_id}] joined rendezvous")
        except Exception as exc:
            logger.warning(
                f"[agent-{self.agent_id}] rendezvous failed (non-fatal): {exc}"
            )

    def _run_loop(self) -> None:
        deadline = time.monotonic() + self.run_duration

        while time.monotonic() < deadline:
            self._send_heartbeat()

            if self.failure_spec is not None:
                beats_done = self.result.heartbeat_count
                if beats_done > self.failure_spec.after_heartbeats:
                    self._report_failure()
                    return

            time.sleep(self.heartbeat_interval)

        # Survived the full duration without being asked to fail.
        self._report_success()

    def _send_heartbeat(self) -> None:
        try:
            self._client.report_heart_beat(int(time.time()))
            self.result.heartbeat_count += 1
        except Exception as exc:
            logger.warning(
                f"[agent-{self.agent_id}] heartbeat error (skipped): {exc}"
            )

    def _report_failure(self) -> None:
        assert self.failure_spec is not None
        spec = self.failure_spec
        try:
            self._client.report_failures(
                error_data=spec.error_data,
                restart_count=0,
                level=spec.level,
            )
        except Exception as exc:
            logger.warning(
                f"[agent-{self.agent_id}] failure report error: {exc}"
            )
        logger.debug(
            f"[agent-{self.agent_id}] reported failure "
            f"(payload {len(spec.error_data)} bytes)"
        )
        self.result.outcome = AgentOutcome.FAILED
        self.result.error = spec.error_data

    def _report_success(self) -> None:
        try:
            self._client.report_succeeded_exited()
        except Exception as exc:
            logger.warning(
                f"[agent-{self.agent_id}] success report error: {exc}"
            )
        logger.debug(f"[agent-{self.agent_id}] reported success")
        self.result.outcome = AgentOutcome.SUCCEEDED


# ---------------------------------------------------------------------------
# Fleet helpers
# ---------------------------------------------------------------------------


def run_agents(
    agents: List[TestTrainingAgent],
) -> List[AgentResult]:
    """Run a list of agents concurrently on daemon threads.

    Returns all results in the same order as ``agents``, after every thread
    has finished.
    """
    results: List[Optional[AgentResult]] = [None] * len(agents)

    def _run(idx: int, agent: TestTrainingAgent) -> None:
        results[idx] = agent.run()

    threads = [
        threading.Thread(
            target=_run,
            args=(i, agent),
            name=f"test-agent-{agent.agent_id}",
            daemon=True,
        )
        for i, agent in enumerate(agents)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return [r for r in results if r is not None]


def make_agents(
    master_addr: str,
    num_agents: int,
    failure_spec: Optional[FailureSpec] = None,
    heartbeat_interval: float = 1.0,
    run_duration: float = 5.0,
) -> List[TestTrainingAgent]:
    """Convenience factory: create ``num_agents`` agents all sharing the same
    ``failure_spec`` (or ``None`` for well-behaved agents that always succeed).
    """
    return [
        TestTrainingAgent(
            agent_id=i,
            master_addr=master_addr,
            num_agents=num_agents,
            failure_spec=failure_spec,
            heartbeat_interval=heartbeat_interval,
            run_duration=run_duration,
        )
        for i in range(num_agents)
    ]


def make_mixed_agents(
    master_addr: str,
    num_agents: int,
    fail_indices: List[int],
    failure_spec: FailureSpec,
    heartbeat_interval: float = 1.0,
    run_duration: float = 5.0,
) -> List[TestTrainingAgent]:
    """Create agents where only those at ``fail_indices`` report failures.

    All other agents behave normally and report success.
    """
    fail_set = set(fail_indices)
    return [
        TestTrainingAgent(
            agent_id=i,
            master_addr=master_addr,
            num_agents=num_agents,
            failure_spec=failure_spec if i in fail_set else None,
            heartbeat_interval=heartbeat_interval,
            run_duration=run_duration,
        )
        for i in range(num_agents)
    ]
