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
from unittest import mock

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.job_auto_scaler import (
    AllreduceTrainingAutoScaler,
    PSTrainingAutoScaler,
)
from dlrover.python.master.node.job_manager import create_job_manager
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.tests.test_utils import (
    MockK8sAllreduceJobArgs,
    MockK8sPSJobArgs,
    mock_k8s_client,
)


class JobAutoScalerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_execute_job_optimization_plan(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()

        manager._scaler.scale = mock.MagicMock(return_value=True)

        auto_scaler = PSTrainingAutoScaler(
            manager._job_resource,
            manager._job_nodes,
            manager._job_optimizer,
            manager._speed_monitor,
            manager._ps_manager,
            manager._worker_manager,
            manager._scaler,
        )
        plan = ResourcePlan()
        plan.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            6, NodeResource(4, 4096)
        )
        plan.node_resources["test-edljob-worker-0"] = NodeResource(8, 8192)
        plan.node_resources["test-edljob-ps-1"] = NodeResource(8, 8192)
        auto_scaler._ps_manager._nodes[1].status = NodeStatus.RUNNING
        scale_plan = auto_scaler.execute_job_optimization_plan(plan)
        self.assertEqual(len(manager._ps_manager._nodes), 4)
        self.assertEqual(len(manager._worker_manager._nodes), 7)
        self.assertEqual(len(scale_plan.remove_nodes), 1)
        self.assertEqual(len(scale_plan.launch_nodes), 5)

        ps_addrs = []
        for i in [0, 3, 2]:
            ps_addrs.append("test-edljob-ps-{}.default.svc:2222".format(i))
        self.assertListEqual(scale_plan.ps_addrs, ps_addrs)


class AllreduceAutoScalerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_execute_job_optimization_plan(self):
        params = MockK8sAllreduceJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()

        for worker in manager._job_nodes[NodeType.WORKER].values():
            worker.status = NodeStatus.RUNNING

        manager._scaler.scale = mock.MagicMock(return_value=True)

        auto_scaler = AllreduceTrainingAutoScaler(
            manager._job_resource,
            manager._job_nodes,
            manager._job_optimizer,
            manager._speed_monitor,
            manager._worker_manager,
            manager._scaler,
        )
        alive_num = auto_scaler._get_alive_worker_num()
        self.assertEqual(alive_num, 16)
