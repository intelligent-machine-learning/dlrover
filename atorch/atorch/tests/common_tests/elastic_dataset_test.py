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

import shlex
import subprocess
import time
import unittest

from dlrover.python.elastic_agent.master_client import MasterClient, build_master_client

from atorch.common.util_func import find_free_port
from atorch.data.elastic_dataset import SimpleElasticDataset


def data_process(i):
    return i


class SimpleElasticDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        port = find_free_port()
        command_line = f"python -m dlrover.python.master.main --port {port} --job_name test --platform local"
        args = shlex.split(command_line)
        self._master_proc = subprocess.Popen(args)
        addr = f"localhost:{port}"
        time.sleep(3)  # Wait the master starts.
        MasterClient._instance = build_master_client(addr)
        self.dataset = SimpleElasticDataset(
            name="test",
            data_process_fn=data_process,
            dataset_size=10000,
            batch_size=10,
            epochs=1,
            shuffle=False,
            num_minibatches_per_shard=10,
        )

    def addCleanup(self):
        self._master_proc.kill()

    def test_index_sharding_client(self):
        self.assertEqual(len(self.dataset), 10000)
        i = self.dataset.__getitem__(0)
        self.assertTrue(i == 0)
        self.assertFalse(self.dataset._shard_client._sample_queue.empty())


if __name__ == "__main__":
    unittest.main()
