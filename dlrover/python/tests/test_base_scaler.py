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

import unittest

from dlrover.python.master.scaler.base_scaler import Scaler


class _StubScaler(Scaler):
    """Minimal concrete Scaler for base-class unit testing."""

    def start(self):
        pass

    def scale(self, plan, **kwargs):
        pass


class ScalerFailedNodeIPTest(unittest.TestCase):
    def setUp(self):
        self._scaler = _StubScaler("test-job")

    def test_get_failed_node_ips_empty_initially(self):
        self.assertEqual(self._scaler.get_failed_node_ips(), [])

    def test_add_failed_node_ip_records_and_dedupes(self):
        self._scaler.add_failed_node_ip("10.0.0.2")
        self._scaler.add_failed_node_ip("10.0.0.1")
        self._scaler.add_failed_node_ip("10.0.0.1")

        # Duplicates are dropped and the result is sorted.
        self.assertEqual(
            self._scaler.get_failed_node_ips(), ["10.0.0.1", "10.0.0.2"]
        )

    def test_add_failed_node_ip_ignores_empty_and_none(self):
        self._scaler.add_failed_node_ip("10.0.0.1")
        self._scaler.add_failed_node_ip("")
        self._scaler.add_failed_node_ip(None)

        # Only the non-empty IP is recorded.
        self.assertEqual(self._scaler.get_failed_node_ips(), ["10.0.0.1"])


if __name__ == "__main__":
    unittest.main()
