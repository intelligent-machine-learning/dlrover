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
    NodeStatus,
    NodeType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.diagnosis.common.diagnosis_action import (
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
        self.assertTrue(isinstance(op.recover_actions()[0], NoAction))
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
        self.assertIsNone(op.check_allreduce_job_pending(mock_nodes, 300, 2))

        # with pending
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
        self.assertIsNotNone(
            op.check_allreduce_job_pending(mock_nodes, 300, 2)
        )

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
        self.assertIsNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 2)
        )
        self.assertIsNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 1)
        )

        # with worker0 pending
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
        self.assertIsNotNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 1)
        )
        self.assertIsNotNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 2)
        )

        # with worker1 pending
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
        self.assertIsNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 1)
        )
        self.assertIsNotNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 2)
        )

        # with ps pending
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
        self.assertIsNotNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 1)
        )
        self.assertIsNotNone(
            op.check_ps_job_pending(mock_worker_nodes + mock_ps_nodes, 300, 2)
        )

    def test_check(self):
        op = SchedulingPreCheckOperator()
        job_args = JobArgs("local", "test", "test")
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE

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

        # without pending
        op.check_allreduce_job_pending = mock.MagicMock(return_value=None)
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 0)
        self.assertFalse(result.result_msg)
        self.assertFalse(result.abnormal_nodes)

        # with pending
        op.check_allreduce_job_pending = mock.MagicMock(
            return_value=Node(node_type="worker", node_id=2, status="PENDING")
        )
        result = op.check(job_args=job_args)
        self.assertEqual(result.result, 1)
        self.assertEqual(
            result.result_msg, SchedulingPreCheckOperator.PENDING_TIMEOUT_MSG
        )
        self.assertEqual(result.abnormal_nodes[0].id, 2)

    def test_wait_scheduling_started(self):
        op = SchedulingPreCheckOperator()
        op.wait_scheduling_started(1, 2)


if __name__ == "__main__":
    unittest.main()
