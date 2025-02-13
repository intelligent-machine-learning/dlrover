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

import unittest
from datetime import datetime, timedelta
from unittest import mock

from dlrover.python.common.constants import (
    DistributionStrategy,
    ErrorMonitorConstants,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    JobAbortionAction,
    NoAction,
)
from dlrover.python.master.diagnosis.precheck_operator import (
    NoPreCheckOperator,
    SchedulingPreCheckOperator,
)
from dlrover.python.scheduler.job import JobArgs


class PreCheckOperatorTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic(self):
        op = NoPreCheckOperator()
        self.assertTrue(op.check())
        self.assertTrue(isinstance(op.recover_actions()[0], NoAction))
        self.assertEqual(op.get_retry_interval_secs(), 5)
        self.assertEqual(op.get_retry_times(), 3)
        self.assertTrue(isinstance(op.failed_actions()[0], NoAction))


class SchedulingPreCheckOperatorTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic(self):
        op = SchedulingPreCheckOperator()
        self.assertEqual(op.get_retry_interval_secs(), 60)
        self.assertEqual(op.get_retry_times(), 15 + 1)

        # test recover actions
        actions = op.recover_actions(result_msg="test", abnormal_nodes=[])
        self.assertTrue(isinstance(actions[0], EventAction))
        self.assertEqual(actions[0].event_msg, "test")
        self.assertEqual(
            actions[0].event_action,
            ErrorMonitorConstants.ACTION_WORKER_PENDING,
        )

        # test failed actions
        self.assertTrue(isinstance(op.failed_actions()[0], JobAbortionAction))

    def test_check_allreduce_job_pending(self):
        op = SchedulingPreCheckOperator()

        # without pending
        mock_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_nodes.append(mock_node)
        self.assertEqual(
            op.check_allreduce_job_pending(mock_nodes, 300, 2), (False, None)
        )

        # with pending timeout
        mock_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_nodes.append(mock_node)
        result = op.check_allreduce_job_pending(mock_nodes, 300, 2)
        self.assertTrue(result[0])
        self.assertIsNotNone(result[1])

        # with pending wait
        mock_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-1)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_nodes.append(mock_node)
        result = op.check_allreduce_job_pending(mock_nodes, 300, 2)
        self.assertTrue(result[0])
        self.assertIsNone(result[1])

    def test_check_ps_job_pending(self):
        op = SchedulingPreCheckOperator()

        # without pending
        mock_worker_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_worker_nodes.append(mock_node)
        mock_ps_nodes = []
        for index in range(2):
            mock_node = Node(
                NodeType.PS,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_ps_nodes.append(mock_node)
        self.assertEqual(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 2),
            (False, None),
        )
        self.assertEqual(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 1),
            (False, None),
        )

        # with worker0 pending timeout
        mock_worker_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_worker_nodes.append(mock_node)
        result = op.check_ps_job_pending(
            mock_worker_nodes + mock_ps_nodes, 300, 1
        )
        self.assertTrue(result[0])
        self.assertIsNotNone(result[1])
        op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 2)
        self.assertTrue(result[0])
        self.assertIsNotNone(result[1])

        # with worker1 pending timeout
        mock_worker_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 1:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_worker_nodes.append(mock_node)
        result = op.check_ps_job_pending(
            mock_worker_nodes + mock_ps_nodes, 300, 1
        )
        self.assertFalse(result[0])
        self.assertIsNone(result[1])
        result = op.check_ps_job_pending(
            mock_worker_nodes + mock_ps_nodes, 300, 2
        )
        self.assertTrue(result[0])
        self.assertIsNotNone(result[1])

        # with worker1 pending wait
        mock_worker_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 1:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-1)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_worker_nodes.append(mock_node)
        result = op.check_ps_job_pending(
            mock_worker_nodes + mock_ps_nodes, 300, 2
        )
        self.assertTrue(result[0])
        self.assertIsNone(result[1])

        # with ps pending timeout
        mock_worker_nodes = []
        for index in range(4):
            mock_node = Node(
                NodeType.WORKER,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            mock_worker_nodes.append(mock_node)
        mock_ps_nodes = []
        for index in range(2):
            mock_node = Node(
                NodeType.PS,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 1:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-10)
            mock_ps_nodes.append(mock_node)
        result = op.check_ps_job_pending(
            mock_worker_nodes + mock_ps_nodes, 300, 1
        )
        self.assertTrue(result[0])
        self.assertIsNotNone(result[1])
        result = op.check_ps_job_pending(
            mock_worker_nodes + mock_ps_nodes, 300, 2
        )
        self.assertTrue(result[0])
        self.assertIsNotNone(result[1])

    def test_check(self):
        op = SchedulingPreCheckOperator()
        job_args = JobArgs("local", "test", "test")
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE
        op.wait_scheduling_started = mock.MagicMock(return_value=True)

        # with timeout = 0
        _dlrover_context = Context.singleton_instance()
        _dlrover_context.seconds_to_wait_pending_pod = 0
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 0)
        self.assertTrue(result.result_msg)
        self.assertFalse(result.abnormal_nodes)
        _dlrover_context.seconds_to_wait_pending_pod = 900

        # with strategy 0
        _dlrover_context.pending_fail_strategy = 0
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 0)
        self.assertTrue(result.result_msg)
        self.assertFalse(result.abnormal_nodes)
        _dlrover_context.pending_fail_strategy = 2

        # allreduce without pending
        op.check_allreduce_job_pending = mock.MagicMock(
            return_value=(False, None)
        )
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 0)
        self.assertFalse(result.result_msg)
        self.assertFalse(result.abnormal_nodes)

        # allreduce with pending timeout
        op.check_allreduce_job_pending = mock.MagicMock(
            return_value=(
                True,
                Node(node_type="worker", node_id=2, status="PENDING"),
            )
        )
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 1)
        self.assertEqual(
            result.result_msg, SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG
        )
        self.assertEqual(result.abnormal_nodes[0].id, 2)

        # allreduce with pending wait
        op.check_allreduce_job_pending = mock.MagicMock(
            return_value=(True, None)
        )
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 1)
        self.assertEqual(
            result.result_msg, SchedulingPreCheckOperator.PENDING_WAIT_MSG
        )
        self.assertFalse(result.abnormal_nodes)

        # ps without pending
        job_args.distribution_strategy = DistributionStrategy.PS
        op.check_ps_job_pending = mock.MagicMock(return_value=(False, None))
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 0)
        self.assertFalse(result.result_msg)
        self.assertFalse(result.abnormal_nodes)

        # ps with pending timeout
        op.check_ps_job_pending = mock.MagicMock(
            return_value=(
                True,
                Node(node_type="worker", node_id=3, status="PENDING"),
            )
        )
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 1)
        self.assertEqual(
            result.result_msg, SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG
        )
        self.assertEqual(result.abnormal_nodes[0].id, 3)

        # other type
        job_args.distribution_strategy = "test"
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 0)
        self.assertTrue("test" in result.result_msg)
        self.assertFalse(result.abnormal_nodes)

        # wait scheduling failed
        op.wait_scheduling_started = mock.MagicMock(return_value=False)
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 1)
        self.assertEqual(
            result.result_msg, SchedulingPreCheckOperator.SCHEDULING_FAILED_MSG
        )
        self.assertFalse(result.abnormal_nodes)

    def test_wait_scheduling_started(self):
        op = SchedulingPreCheckOperator()
        self.assertFalse(op.wait_scheduling_started(1, 2))


if __name__ == "__main__":
    unittest.main()
