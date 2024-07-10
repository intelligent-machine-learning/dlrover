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

import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.dist_job_manager import create_job_manager
from dlrover.python.master.node.job_auto_scaler import (
    AllreduceTrainingAutoScaler,
    PSTrainingAutoScaler,
)
from dlrover.python.master.resource.optimizer import ResourcePlan
from dlrover.python.tests.test_utils import (
    MockK8sAllreduceJobArgs,
    MockK8sPSJobArgs,
    mock_k8s_client,
)

_dlrover_context = Context.singleton_instance()


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
        plan.node_resources["test-edljob-worker-1"] = NodeResource(8, 8192)
        plan.node_resources["test-edljob-ps-1"] = NodeResource(8, 8192)
        auto_scaler._ps_manager._nodes[1].status = NodeStatus.RUNNING
        auto_scaler._worker_manager._nodes[0].critical = True
        scale_plan = auto_scaler.execute_job_optimization_plan(plan)
        self.assertEqual(len(manager._ps_manager._nodes), 4)
        self.assertEqual(len(manager._worker_manager._nodes), 7)
        self.assertEqual(len(scale_plan.remove_nodes), 1)
        self.assertEqual(len(scale_plan.launch_nodes), 5)
        remove_node = scale_plan.remove_nodes[0]
        self.assertTrue(remove_node.migrated)
        self.assertTrue(remove_node.is_released)
        self.assertFalse(remove_node.relaunchable)
        self.assertEqual(remove_node.name, "test-edljob-worker-1")

        ps_addrs = []
        for i in [0, 3, 2]:
            ps_addrs.append("test-edljob-ps-{}.default.svc:2222".format(i))
        self.assertListEqual(scale_plan.ps_addrs, ps_addrs)

        plan = None
        scale_plan = auto_scaler.execute_job_optimization_plan(plan)
        self.assertIsNotNone(scale_plan)

        # mock for following async thread testing
        auto_scaler._speed_monitor.worker_adjustment_finished = mock.MagicMock(
            side_effect=Exception()
        )
        _dlrover_context = Context.singleton_instance()
        _dlrover_context.auto_ps_enabled = True
        _dlrover_context.auto_worker_enabled = True

        auto_scaler.start_auto_scaling()
        time.sleep(3)
        self.assertTrue(auto_scaler._autoscaling_started)
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("ps-autoscaler", active_threads_name)

        auto_scaler.stop_auto_scaling()
        self.assertFalse(auto_scaler._autoscaling_started)

    def test_reduce_timeout_pending_node_resource(self):
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
        auto_scaler._autoscaling_started = True
        ps0 = manager._ps_manager._nodes[0]
        ps0.config_resource.cpu = 16
        ps0.status = NodeStatus.PENDING
        ps0.create_time = datetime.now() + timedelta(days=-1)
        plan = auto_scaler._reduce_timeout_pending_node_resource()
        self.assertEqual(
            plan.ps_addrs,
            [
                "test-edljob-ps-0.default.svc:2222",
                "test-edljob-ps-1.default.svc:2222",
                "test-edljob-ps-2.default.svc:2222",
            ],
        )


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

        plan = None
        scale_plan = auto_scaler.execute_job_optimization_plan(plan)
        self.assertIsNotNone(scale_plan)

        # mock for following async thread testing
        auto_scaler._scale_interval = 1
        auto_scaler._get_alive_worker_num = mock.MagicMock(
            side_effect=Exception()
        )
        _dlrover_context = Context.singleton_instance()
        _dlrover_context.auto_worker_enabled = True

        auto_scaler.start_auto_scaling()
        time.sleep(3)
        self.assertTrue(auto_scaler._autoscaling_started)
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("allreduce-autoscaler", active_threads_name)

        auto_scaler.stop_auto_scaling()
        self.assertFalse(auto_scaler._autoscaling_started)
