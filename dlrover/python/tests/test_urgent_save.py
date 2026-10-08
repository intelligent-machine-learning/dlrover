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

import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from dlrover.python.common.urgent_save import (
    KV_CMD_KEY,
    KV_CLEANUP_KEY,
    STATUS_ERROR,
    STATUS_RECEIVED,
    STATUS_SAVED,
    atomic_write_marker,
    build_cleanup_cmd,
    build_cmd,
    build_result,
    marker_path,
    parse_payload,
    read_tracker,
    register_master_kv_store,
    remove_marker,
    result_key,
    tracker_path,
)
from dlrover.python.elastic_agent.torch.urgent_save_listener import (
    UrgentSaveListener,
)


class _FakeMasterClient:
    """A minimal master client over a dict, mimicking kv_store_get/set."""

    def __init__(self):
        self.store = {}

    def kv_store_get(self, key):
        return self.store.get(key, b"")

    def kv_store_set(self, key, value):
        self.store[key] = value
        return True


def _write_tracker(save_dir, iteration):
    with open(tracker_path(save_dir), "w") as f:
        f.write(str(iteration))


class UrgentSaveProtocolTest(unittest.TestCase):
    def setUp(self):
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_paths(self):
        self.assertEqual(
            marker_path(self.save_dir),
            os.path.join(self.save_dir, ".urgent_save.marker"),
        )
        self.assertEqual(
            tracker_path(self.save_dir),
            os.path.join(self.save_dir, "latest_checkpointed_iteration.txt"),
        )

    def test_payload_roundtrip(self):
        cmd = parse_payload(build_cmd(1.5, self.save_dir, "node failure"))
        self.assertEqual(cmd["action"], "save")
        self.assertEqual(cmd["ts"], 1.5)
        self.assertEqual(cmd["save_dir"], self.save_dir)
        self.assertEqual(cmd["reason"], "node failure")

        result = parse_payload(build_result(1.5, STATUS_SAVED, "done"))
        self.assertEqual(result["cmd_ts"], 1.5)
        self.assertEqual(result["status"], STATUS_SAVED)

    def test_parse_payload_invalid(self):
        self.assertIsNone(parse_payload(b""))
        self.assertIsNone(parse_payload(b"not-json{"))
        self.assertIsNone(parse_payload(b"[1, 2]"))  # not a JSON object

    def test_atomic_write_and_remove_marker(self):
        path = atomic_write_marker(self.save_dir, b'{"v":1}')
        self.assertEqual(path, marker_path(self.save_dir))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b'{"v":1}')
        # No tmp leftovers next to the marker.
        leftovers = [
            f
            for f in os.listdir(self.save_dir)
            if f.startswith(".urgent_save.marker.tmp")
        ]
        self.assertEqual(leftovers, [])
        self.assertTrue(remove_marker(self.save_dir))
        self.assertFalse(os.path.exists(path))
        # Idempotent.
        self.assertFalse(remove_marker(self.save_dir))

    def test_read_tracker(self):
        # Missing tracker: the job has not saved any checkpoint yet.
        self.assertEqual(read_tracker(self.save_dir), (None, 0.0))
        _write_tracker(self.save_dir, 100)
        iteration, mtime = read_tracker(self.save_dir)
        self.assertEqual(iteration, 100)
        self.assertGreater(mtime, 0.0)

    def test_kv_store_registry(self):
        kv = object()
        register_master_kv_store(kv)
        from dlrover.python.common.urgent_save import get_master_kv_store

        self.assertIs(get_master_kv_store(), kv)


class UrgentSaveListenerTest(unittest.TestCase):
    def setUp(self):
        self.save_dir = tempfile.mkdtemp()
        self.client = _FakeMasterClient()
        self.listener = UrgentSaveListener(
            self.client, poll_interval_seconds=0.01
        )

    def tearDown(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_empty_bus_is_noop(self):
        self.listener._poll_once()
        self.assertFalse(os.path.exists(marker_path(self.save_dir)))
        self.assertEqual(self.client.store, {})

    def test_cleanup_removes_stale_marker(self):
        atomic_write_marker(self.save_dir, b'{"v":1,"stale":true}')
        self.client.store[KV_CLEANUP_KEY] = build_cleanup_cmd(
            1.0, self.save_dir
        )
        self.listener._poll_once()
        self.assertFalse(os.path.exists(marker_path(self.save_dir)))
        # The same cleanup ts is not processed twice.
        atomic_write_marker(self.save_dir, b'{"again":true}')
        self.listener._poll_once()
        self.assertTrue(os.path.exists(marker_path(self.save_dir)))

    def test_save_request_writes_marker_and_reports_received(self):
        self.client.store[KV_CMD_KEY] = build_cmd(2.0, self.save_dir, "fail")
        self.listener._poll_once()
        marker = marker_path(self.save_dir)
        self.assertTrue(os.path.exists(marker))
        payload = parse_payload(self.client.store[result_key(2.0)])
        self.assertEqual(payload["status"], STATUS_RECEIVED)
        # The same cmd ts is deduplicated (no result rewrite).
        self.listener._poll_once()
        self.assertEqual(
            self.client.store[result_key(2.0)],
            build_result(2.0, STATUS_RECEIVED, "marker written at " + marker),
        )

    def test_save_request_without_save_dir_reports_error(self):
        empty_dir = json.dumps(
            {
                "v": 1,
                "ts": 3.0,
                "action": "save",
                "save_dir": "",
                "reason": "",
            }
        ).encode("utf-8")
        self.client.store[KV_CMD_KEY] = empty_dir
        self.listener._poll_once()
        payload = parse_payload(self.client.store[result_key(3.0)])
        self.assertEqual(payload["status"], STATUS_ERROR)

    def test_observe_reports_saved_when_tracker_flips(self):
        _write_tracker(self.save_dir, 10)
        marker = atomic_write_marker(self.save_dir, b'{"v":1}')
        observer = threading.Thread(
            target=self.listener._observe,
            args=(4.0, self.save_dir, 10, marker),
            daemon=True,
        )
        observer.start()
        # Give the observer a few polls on the un-flipped tracker first.
        time.sleep(0.05)
        self.assertTrue(observer.is_alive())
        # The training saved and finalized: the tracker flips late enough
        # for the mtime comparison to hold.
        time.sleep(0.02)
        _write_tracker(self.save_dir, 11)
        observer.join(timeout=5)
        self.assertFalse(observer.is_alive())
        payload = parse_payload(self.client.store[result_key(4.0)])
        self.assertEqual(payload["status"], STATUS_SAVED)
        self.assertIn("11", payload["message"])

    def test_observe_waits_when_tracker_does_not_flip(self):
        _write_tracker(self.save_dir, 10)
        marker = atomic_write_marker(self.save_dir, b'{"v":1}')
        observer = threading.Thread(
            target=self.listener._observe,
            args=(5.0, self.save_dir, 10, marker),
            daemon=True,
        )
        observer.start()
        time.sleep(0.1)
        # No flip: the observer keeps waiting, no result is published.
        self.assertTrue(observer.is_alive())
        self.assertNotIn(result_key(5.0), self.client.store)
        _write_tracker(self.save_dir, 10)  # same iteration is not a flip
        time.sleep(0.1)
        self.assertTrue(observer.is_alive())
        self.assertNotIn(result_key(5.0), self.client.store)

    def test_observe_reports_first_flip_when_base_tracker_missing(self):
        # No checkpoint had been saved before the request: the first tracker
        # appearance (after the marker) counts as a flip.
        marker = atomic_write_marker(self.save_dir, b'{"v":1}')
        observer = threading.Thread(
            target=self.listener._observe,
            args=(6.0, self.save_dir, None, marker),
            daemon=True,
        )
        observer.start()
        time.sleep(0.02)
        _write_tracker(self.save_dir, 1)
        observer.join(timeout=5)
        self.assertFalse(observer.is_alive())
        payload = parse_payload(self.client.store[result_key(6.0)])
        self.assertEqual(payload["status"], STATUS_SAVED)


if __name__ == "__main__":
    unittest.main()
