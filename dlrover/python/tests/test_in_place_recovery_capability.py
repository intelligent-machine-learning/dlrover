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

from pathlib import Path
import shutil
import tempfile
import unittest

from dlrover.python.elastic_agent.in_place_recovery_capability import (
    CapabilityRegistrationError,
    build_process_capability,
    new_process_id,
    read_process_capabilities,
    read_process_capability,
    register_process_capability,
    unregister_process_capability,
)
from dlrover.python.elastic_agent.in_place_recovery_files import (
    atomic_write_json,
    ensure_recovery_layout,
)
from dlrover.python.elastic_agent.in_place_recovery_protocol import (
    FEATURE_CHECKPOINT_RESTORE,
    FEATURE_PROCESS_GROUP_RECONFIGURE,
    FEATURE_RECOVERY_COMMAND,
    RecoveryFileLayout,
    RecoveryProtocolError,
)


ALL_FEATURES = (
    FEATURE_CHECKPOINT_RESTORE,
    FEATURE_PROCESS_GROUP_RECONFIGURE,
    FEATURE_RECOVERY_COMMAND,
)


class InPlaceRecoveryCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dlrover-capability-test-"))
        self.layout = RecoveryFileLayout(self.temp_dir / "job")
        ensure_recovery_layout(self.layout)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _capability(
        self,
        rank=0,
        generation=0,
        process_id=None,
        pid=101,
        run_id="job-1",
        features=ALL_FEATURES,
    ):
        return build_process_capability(
            run_id=run_id,
            generation=generation,
            logical_rank=rank,
            features=features,
            pid=pid,
            process_id=process_id or f"process-{rank}-{generation}-{pid}",
        )

    def test_register_and_read_capability(self):
        capability = self._capability(rank=3)
        path = register_process_capability(
            self.layout,
            capability,
            required_features=ALL_FEATURES,
        )

        self.assertEqual(path, self.layout.capabilities_path(3))
        self.assertEqual(read_process_capability(self.layout, 3), capability)

        # Repeating registration for the same process is safe.
        self.assertEqual(
            register_process_capability(self.layout, capability), path
        )

    def test_generation_and_identity_conflicts_are_rejected(self):
        initial = self._capability(rank=2, generation=0, pid=101)
        register_process_capability(self.layout, initial)

        with self.assertRaises(CapabilityRegistrationError):
            register_process_capability(
                self.layout,
                self._capability(rank=2, generation=0, pid=202),
            )
        with self.assertRaises(CapabilityRegistrationError):
            register_process_capability(
                self.layout,
                self._capability(rank=2, generation=0, run_id="other-job"),
            )

        replacement = self._capability(rank=2, generation=1, pid=202)
        register_process_capability(self.layout, replacement)
        self.assertEqual(read_process_capability(self.layout, 2), replacement)

        with self.assertRaises(CapabilityRegistrationError):
            register_process_capability(self.layout, initial)

    def test_required_features_are_checked_before_publish(self):
        capability = self._capability(features=(FEATURE_RECOVERY_COMMAND,))
        with self.assertRaises(CapabilityRegistrationError):
            register_process_capability(
                self.layout,
                capability,
                required_features=ALL_FEATURES,
            )
        self.assertIsNone(read_process_capability(self.layout, 0))

        with self.assertRaises(CapabilityRegistrationError):
            register_process_capability(
                self.layout,
                self._capability(),
                required_features="not-a-feature-list",
            )

    def test_discovery_is_rank_sorted_and_filterable(self):
        capabilities = (
            self._capability(rank=12, generation=2),
            self._capability(rank=1, generation=1),
            self._capability(rank=3, generation=1),
        )
        for capability in capabilities:
            register_process_capability(self.layout, capability)

        discovered = read_process_capabilities(self.layout, run_id="job-1")
        self.assertEqual(
            tuple(item.logical_rank for item in discovered), (1, 3, 12)
        )
        generation_one = read_process_capabilities(
            self.layout, run_id="job-1", generation=1
        )
        self.assertEqual(
            tuple(item.logical_rank for item in generation_one), (1, 3)
        )

    def test_unregister_cannot_remove_replacement(self):
        original = self._capability(rank=4, generation=0)
        replacement = self._capability(rank=4, generation=1, pid=202)
        register_process_capability(self.layout, original)
        register_process_capability(self.layout, replacement)

        self.assertFalse(unregister_process_capability(self.layout, original))
        self.assertEqual(read_process_capability(self.layout, 4), replacement)
        self.assertTrue(unregister_process_capability(self.layout, replacement))
        self.assertIsNone(read_process_capability(self.layout, 4))

    def test_path_payload_rank_mismatch_is_rejected(self):
        payload = self._capability(rank=2).to_dict()
        atomic_write_json(self.layout.capabilities_path(5), payload)
        with self.assertRaises(RecoveryProtocolError):
            read_process_capability(self.layout, 5)

    def test_process_ids_are_unique(self):
        self.assertNotEqual(new_process_id(pid=1), new_process_id(pid=1))


if __name__ == "__main__":
    unittest.main()

