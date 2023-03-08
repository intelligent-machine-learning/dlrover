# Copyright 2022 The EasyDL Authors. All rights reserved.
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

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeType,
    PlatformType,
)
from dlrover.python.scheduler.kubernetes import K8sJobArgs
from dlrover.python.tests.test_utils import mock_k8s_client


class k8sJobArgsTest(unittest.TestCase):
    def test_initialize_params(self):
        mock_k8s_client()
        params = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
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
        self.assertEqual(worker_params.group_resource.node_resource.cpu, 0)
        self.assertEqual(worker_params.group_resource.node_resource.memory, 0)

        ps_params = params.node_args[NodeType.PS]
        self.assertEqual(ps_params.restart_count, 3)
        self.assertEqual(ps_params.restart_timeout, 0)
        self.assertEqual(ps_params.group_resource.count, 3)
        self.assertEqual(ps_params.group_resource.node_resource.cpu, 1)
        self.assertEqual(ps_params.group_resource.node_resource.memory, 4096)
        self.assertEqual(
            ps_params.group_resource.node_resource.priority, "high"
        )
