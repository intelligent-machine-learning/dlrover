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

import json
from pathlib import Path
import unittest

from dlrover.python.elastic_agent.in_place_recovery_protocol import (
    FEATURE_CHECKPOINT_RESTORE,
    FEATURE_PROCESS_GROUP_RECONFIGURE,
    FEATURE_RECOVERY_COMMAND,
    RecoveryCapability,
    RecoveryFileLayout,
    RecoveryHeartbeat,
    RecoveryProtocolError,
    RecoveryReplacement,
    RecoveryRequest,
    RecoveryStatus,
    STATUS_ABORTED,
    STATUS_FAILED,
    STATUS_OBSERVED,
    STATUS_QUIESCED,
    STATUS_READY,
    STATUS_REGISTERED,
    is_valid_status_transition,
)


class InPlaceRecoveryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.replacements = (
            RecoveryReplacement(2, "replacement-g7-rank2"),
            RecoveryReplacement(5, "replacement-g7-rank5"),
        )
        self.request = RecoveryRequest(
            command_id="generation-00000007",
            run_id="job-123",
            generation=7,
            failed_logical_ranks=(5, 2),
            replacements=self.replacements,
            checkpoint_id="iter_0001200",
            topology_digest="sha256:topology",
            deadline_ms=1770000123456,
        )

    def test_request_round_trip_is_canonical_and_digestable(self):
        payload = self.request.to_json()
        self.assertEqual(payload, self.request.to_json())
        self.assertEqual(payload, payload.encode("utf-8").decode("utf-8"))

        restored = RecoveryRequest.from_json(payload)
        self.assertEqual(restored, self.request)
        self.assertEqual(restored.failed_logical_ranks, (2, 5))
        self.assertEqual(
            tuple(item.logical_rank for item in restored.replacements), (2, 5)
        )
        self.assertEqual(len(restored.digest), 64)

    def test_capability_status_and_heartbeat_round_trip(self):
        capability = RecoveryCapability(
            run_id="job-123",
            generation=0,
            logical_rank=5,
            pid=12345,
            process_id="rank5-process-a",
            features=(
                FEATURE_CHECKPOINT_RESTORE,
                FEATURE_PROCESS_GROUP_RECONFIGURE,
                FEATURE_RECOVERY_COMMAND,
            ),
        )
        self.assertEqual(
            RecoveryCapability.from_json(capability.to_json()), capability
        )

        status = RecoveryStatus(
            command_id=self.request.command_id,
            run_id="job-123",
            generation=7,
            logical_rank=5,
            pid=12345,
            process_id="rank5-process-a",
            state=STATUS_QUIESCED,
            timestamp_ms=1770000123000,
        )
        self.assertEqual(RecoveryStatus.from_json(status.to_json()), status)

        heartbeat = RecoveryHeartbeat(
            run_id="job-123",
            generation=7,
            logical_rank=5,
            pid=12345,
            process_id="rank5-process-a",
            timestamp_ms=1770000123000,
        )
        self.assertEqual(RecoveryHeartbeat.from_json(heartbeat.to_json()), heartbeat)

    def test_paths_are_deterministic_and_rank_scoped(self):
        layout = RecoveryFileLayout(Path("/tmp/dlrover/job-123"))
        self.assertEqual(
            layout.capabilities_path(5),
            Path("/tmp/dlrover/job-123/capabilities/rank-000005.json"),
        )
        self.assertEqual(
            layout.request_path(7),
            Path("/tmp/dlrover/job-123/requests/generation-00000007.json"),
        )
        self.assertEqual(
            layout.status_path(5),
            Path("/tmp/dlrover/job-123/status/rank-000005.json"),
        )
        self.assertEqual(
            layout.heartbeat_path(5),
            Path("/tmp/dlrover/job-123/heartbeats/rank-000005.json"),
        )
        self.assertEqual(len(layout.directories), 5)

    def test_status_transitions_are_monotonic(self):
        self.assertTrue(is_valid_status_transition(None, STATUS_REGISTERED))
        self.assertTrue(is_valid_status_transition(STATUS_REGISTERED, STATUS_OBSERVED))
        self.assertTrue(is_valid_status_transition(STATUS_OBSERVED, STATUS_ABORTED))
        self.assertTrue(is_valid_status_transition(STATUS_ABORTED, STATUS_QUIESCED))
        self.assertFalse(is_valid_status_transition(STATUS_QUIESCED, STATUS_READY))
        self.assertTrue(is_valid_status_transition(STATUS_QUIESCED, STATUS_FAILED))
        self.assertFalse(is_valid_status_transition(STATUS_READY, STATUS_OBSERVED))
        self.assertFalse(is_valid_status_transition(STATUS_FAILED, STATUS_READY))

    def test_invalid_payloads_are_rejected(self):
        with self.assertRaises(RecoveryProtocolError):
            RecoveryRequest.from_json(self.request.to_json().replace(
                '"protocol_version":1', '"protocol_version":99'
            ))

        invalid = json.loads(self.request.to_json())
        invalid["replacements"] = [
            {"logical_rank": 2, "replacement_id": "replacement-g7-rank2"}
        ]
        with self.assertRaises(RecoveryProtocolError):
            RecoveryRequest.from_json(json.dumps(invalid))

        with self.assertRaises(RecoveryProtocolError):
            RecoveryStatus(
                command_id="generation-00000007",
                run_id="job-123",
                generation=7,
                logical_rank=5,
                pid=12345,
                process_id="rank5-process-a",
                state="unknown",
                timestamp_ms=1770000123000,
            )


if __name__ == "__main__":
    unittest.main()

