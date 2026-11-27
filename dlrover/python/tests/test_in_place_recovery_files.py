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

import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import unittest

from dlrover.python.elastic_agent.in_place_recovery_files import (
    RecoveryFileError,
    atomic_write_bytes,
    atomic_write_json,
    cleanup_stale_files,
    ensure_private_directory,
    ensure_recovery_layout,
    read_bytes,
    read_json,
)
from dlrover.python.elastic_agent.in_place_recovery_protocol import (
    RecoveryFileLayout,
    RecoveryProtocolError,
)


class InPlaceRecoveryFilesTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dlrover-recovery-test-"))
        self.layout = RecoveryFileLayout(self.temp_dir / "job")
        ensure_recovery_layout(self.layout)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_layout_is_private_even_below_shared_parent(self):
        self.assertTrue(self.layout.root.is_dir())
        for directory in self.layout.directories:
            mode = stat.S_IMODE(os.stat(directory).st_mode)
            self.assertEqual(mode & 0o077, 0)
            self.assertEqual(mode & 0o700, 0o700)

    def test_atomic_json_round_trip_and_duplicate_write(self):
        path = self.layout.request_path(4)
        payload = {"generation": 4, "state": "published"}
        atomic_write_json(path, payload)
        atomic_write_json(path, payload)

        self.assertEqual(read_json(path), payload)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        self.assertEqual(mode & 0o077, 0)
        self.assertEqual(mode & 0o600, 0o600)
        self.assertEqual(
            tuple(path.parent.iterdir()),
            (path,),
        )

    def test_atomic_bytes_replaces_complete_content(self):
        path = self.layout.status_path(3)
        atomic_write_bytes(path, b"first")
        atomic_write_bytes(path, b"second")
        self.assertEqual(read_bytes(path), b"second")

    def test_missing_and_malformed_documents(self):
        path = self.layout.status_path(3)
        self.assertIsNone(read_json(path))
        atomic_write_bytes(path, b"not-json")
        with self.assertRaises(RecoveryProtocolError):
            read_json(path)

        atomic_write_bytes(path, b"[]")
        with self.assertRaises(RecoveryProtocolError):
            read_json(path)

    def test_private_permissions_and_symlinks_are_required(self):
        insecure_dir = self.temp_dir / "insecure"
        insecure_dir.mkdir(mode=0o755)
        os.chmod(insecure_dir, 0o755)
        with self.assertRaises(RecoveryFileError):
            ensure_private_directory(insecure_dir)

        target = self.layout.status_path(4)
        target.write_text("{}")
        os.chmod(target, 0o644)
        with self.assertRaises(RecoveryFileError):
            read_json(target)

        link = self.layout.status_path(5)
        link.symlink_to(target)
        with self.assertRaises(RecoveryFileError):
            atomic_write_bytes(link, b"{}")

    def test_cleanup_removes_old_generations_and_crashed_temps(self):
        atomic_write_json(self.layout.request_path(1), {"generation": 1})
        atomic_write_json(self.layout.request_path(3), {"generation": 3})
        atomic_write_json(self.layout.status_path(1), {"generation": 1})
        atomic_write_json(self.layout.status_path(3), {"generation": 3})
        atomic_write_json(self.layout.heartbeat_path(2), {"generation": 2})
        atomic_write_json(self.layout.heartbeat_path(3), {"generation": 3})
        atomic_write_json(self.layout.capabilities_path(2), {"generation": 2})
        atomic_write_json(self.layout.capabilities_path(3), {"generation": 3})

        temporary = self.layout.root / "requests" / ".crashed.tmp-test"
        temporary.write_bytes(b"partial")
        old_time = time.time() - 10
        os.utime(temporary, (old_time, old_time))

        removed = cleanup_stale_files(
            self.layout, keep_generation=3, temp_max_age_seconds=0
        )
        removed = {path.name for path in removed}
        self.assertIn("generation-00000001.json", removed)
        self.assertIn("rank-000001.json", removed)
        self.assertIn("rank-000002.json", removed)
        self.assertIn(".crashed.tmp-test", removed)
        self.assertIsNotNone(read_json(self.layout.request_path(3)))
        self.assertIsNotNone(read_json(self.layout.status_path(3)))
        self.assertIsNotNone(read_json(self.layout.heartbeat_path(3)))
        self.assertIsNotNone(read_json(self.layout.capabilities_path(3)))


if __name__ == "__main__":
    unittest.main()

