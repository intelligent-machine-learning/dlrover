# Copyright 2023 The DLRover Authors. All rights reserved.
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

from dlrover.trainer.torch.elastic_run import (
    _check_dlrover_master_available,
    _elastic_config_from_args,
    _launch_dlrover_local_master,
    parse_args,
)


class ElasticRunTest(unittest.TestCase):
    def test_launch_local_master(self):
        handler, addr = _launch_dlrover_local_master()
        available = _check_dlrover_master_available(addr)
        self.assertTrue(available)
        handler.close()

    def test_elastic_config_from_args(self):
        args = [
            "--network_check",
            "--auto_tunning",
            "--node_unit",
            "4",
            "--nnodes",
            "4",
            "test.py",
            "--batch_size",
            "16",
        ]
        args = parse_args(args)
        config, cmd, cmd_args = _elastic_config_from_args(args)
        self.assertTrue(config.network_check)
        self.assertTrue(config.auto_tunning)
        self.assertEqual(config.node_unit, 4)
        self.assertEqual(cmd, "/usr/local/bin/python")
        self.assertListEqual(cmd_args, ["-u", "test.py", "--batch_size", "16"])
