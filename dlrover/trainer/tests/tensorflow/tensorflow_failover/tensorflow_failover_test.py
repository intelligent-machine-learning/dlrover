# Copyright 2022 The DLRover Authors. All rights reserved.
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
import unittest

from dlrover.trainer.tensorflow.failover.tensorflow_failover import (
    TensorflowFailover,
)


class MockFailoverClient:
    def __init__(self, role):
        self._role = role

    def init_version(self):
        return

    def get_training_ps_addr(self):
        return ["web04-pod2.default.svc:5004"], False


class TensorflowFailoverTest(unittest.TestCase):
    def test_ps_address_changed(self):
        cluster = {
            "cluster": {
                "ps": ["web04-pod2.default.svc:5002"],
                "chief": ["p1.default.svc:5001"],
                "worker": ["web04-pod1.default.svc:5000"],
            },
            "task": {"type": "worker", "index": 0},
        }
        os.environ["TF_CONFIG"] = json.dumps(cluster)
        role = "worker:0"
        tensorflow_failover = TensorflowFailover(
            failover_client=MockFailoverClient
        )
        self.assertTrue(tensorflow_failover._role, role)
        self.assertListEqual(
            tensorflow_failover.curr_ps_address, cluster["cluster"]["ps"]
        )
        ps_cluster, t = tensorflow_failover.ps_addresses_changed()
        self.assertEqual(t, "migrating")
        self.assertTrue(ps_cluster, ["web04-pod2.default.svc:5004"])
        tensorflow_failover.refresh_env()
        self.assertTrue(
            "web04-pod2.default.svc:5004" in os.environ["TF_CONFIG"]
        )
        TF_CONFIG = json.loads(os.environ["TF_CONFIG"])
        self.assertListEqual(
            TF_CONFIG["cluster"]["ps"], ["web04-pod2.default.svc:5004"]
        )

    def test_refresh_env(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
