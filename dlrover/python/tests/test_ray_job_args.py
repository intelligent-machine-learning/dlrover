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

import json
import os
import unittest

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeType,
    PlatformType,
)
from dlrover.python.master.stats.stats_backend import parse_yaml_file
from dlrover.python.scheduler.ray import RayJobArgs


class RayJobArgsTest(unittest.TestCase):
    def test_initialize_params(self):
        file = "data/demo.yaml"
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)
        data = parse_yaml_file(file_path)
        with open("test.json", "w") as f:
            json.dump(data, f)

        params = RayJobArgs(PlatformType.RAY, "default", "test")
        self.assertEqual(params.job_name, "test")
        self.assertEqual(params.namespace, "default")
        params.initilize()
        self.assertTrue(NodeType.WORKER in params.node_args)
        self.assertTrue(NodeType.PS in params.node_args)
        self.assertTrue(
            params.distribution_strategy == DistributionStrategy.PS
        )
        worker_params = params.node_args[NodeType.WORKER]
        self.assertEqual(worker_params.restart_count, 3)
        self.assertEqual(worker_params.restart_timeout, 0)
        self.assertEqual(worker_params.group_resource.count, 0)
