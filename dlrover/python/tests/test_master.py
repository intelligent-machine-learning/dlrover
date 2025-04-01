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
from unittest.mock import MagicMock, patch

from dlrover.python.common.constants import (
    DistributionStrategy,
    JobExitReason,
    JobStage,
    NodeStatus,
    NodeType,
    PreCheckStatus,
    RendezvousName,
)
from dlrover.python.common.global_context import Context
from dlrover.python.diagnosis.common.diagnosis_action import (
    JobAbortionAction,
    NoAction,
)
from dlrover.python.elastic_agent.master_client import build_master_client
from dlrover.python.master.args import parse_master_args
from dlrover.python.master.dist_master import (
    DistributedJobMaster,
    _create_master_service_on_k8s,
)
from dlrover.python.master.main import run, update_context
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.shard.dataset_splitter import new_dataset_splitter
from dlrover.python.tests.test_utils import (
    MockK8sPSJobArgs,
    mock_k8s_client,
    start_local_master,
)

_dlrover_context = Context.singleton_instance()


class DistributedJobMasterTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        params = MockK8sPSJobArgs()
        params.initilize()
        self.master = DistributedJobMaster(2222, params)
        self.job_context = get_job_context()
        self.job_context.update_job_stage(JobStage.JOB_INIT)

    def tearDown(self):
        self.job_context.clear_job_nodes()
        self.job_context.request_stop()

    def test_exit_by_workers(self):
        self.master.job_manager._init_nodes()
        job_nodes = self.job_context.job_nodes()
        for node in job_nodes[NodeType.WORKER].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.EVALUATOR].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.CHIEF].values():
            node.status = NodeStatus.FINISHED
        self.job_context.update_job_nodes(job_nodes)
        self.master.run()
        self.assertEqual(self.master._exit_code, 0)
        self.assertEqual(self.master._exit_reason, JobExitReason.SUCCEEDED)

    def test_exit_by_tasks(self):
        self.master.job_manager._init_nodes()
        job_nodes = self.job_context.job_nodes()
        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.EVALUATOR].values():
            node.status = NodeStatus.FINISHED
        for node in job_nodes[NodeType.CHIEF].values():
            node.status = NodeStatus.FINISHED

        job_nodes[NodeType.WORKER][0].status = NodeStatus.FINISHED
        self.job_context.update_job_nodes(job_nodes)

        splitter = new_dataset_splitter(
            False,
            100,
            10000,
            1,
            "test",
            "table",
        )

        self.master.task_manager.new_dataset(10, 10000, "test", splitter)

        for dataset in self.master.task_manager._datasets.values():
            dataset.todo.clear()
            dataset.doing.clear()
            dataset._dataset_splitter.epoch = 10
        self.master.run()
        self.assertEqual(self.master._exit_code, 0)
        self.assertEqual(self.master._exit_reason, JobExitReason.SUCCEEDED)

    def test_early_stop(self):
        self.master.job_manager._init_nodes()
        job_nodes = self.job_context.job_nodes()

        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
            node.is_recovered_oom = True
            node.create_time = datetime.now() + timedelta(days=-1)
        self.job_context.update_job_nodes(job_nodes)

        exit_code = self.master.run()
        self.master.job_manager.clear_all_nodes()
        self.assertEqual(exit_code, 1)

    def test_update_context(self):
        job_args = MockK8sPSJobArgs()
        job_args.initilize()
        job_args.relaunch_always = False
        job_args.distribution_strategy = DistributionStrategy.ALLREDUCE
        update_context(job_args)
        self.assertTrue(_dlrover_context.relaunch_always)
        self.assertTrue(_dlrover_context.auto_ps_enabled)
        self.assertTrue(_dlrover_context.auto_worker_enabled)

    @patch("dlrover.python.master.local_master.LocalJobMaster.__init__")
    def test_run(self, mock_master_init):
        # interrupt when init master(no need for the test)
        mock_master_init.side_effect = Exception("test")

        original_args = [
            "--platform",
            "local",
            "--job_name",
            "test",
            "--namespace",
            "default",
        ]

        try:
            run(parse_master_args(original_args))
        except Exception:
            pass

    def test_create_master_service_on_k8s(self):
        succeed = _create_master_service_on_k8s(
            "dlrover", "test", "12345", 12345
        )
        self.assertTrue(succeed)

    @patch(
        "dlrover.python.master.dist_master.DistributedJobMaster.request_stop"
    )
    def test_diagnose_job(self, mock_request_stop):
        # close thread in 3 secs
        def stop_master():
            time.sleep(3)
            get_job_context().request_stop()

        threading.Thread(target=stop_master).start()

        # test actions
        get_job_context().enqueue_diagnosis_action(JobAbortionAction())
        get_job_context().enqueue_diagnosis_action(NoAction())
        self.master._diagnose_job()
        mock_request_stop.assert_called_once()

    def test_pre_check(self):
        self.master.diagnosis_manager.pre_check = MagicMock(return_value=True)
        self.master.pre_check()


class LocalJobMasterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        self.master_client = build_master_client(addr, 0.5)
        self.job_context = get_job_context()
        self.job_context._pre_check_status = PreCheckStatus.CHECKING

    def tearDown(self):
        self._master.stop()
        self.job_context.clear_job_nodes()
        self.job_context.request_stop()
        self.job_context._pre_check_status = PreCheckStatus.CHECKING

    def test_task_manager(self):
        self.master_client.report_dataset_shard_params(
            64, 1, 10000, False, 10, "test-ds"
        )
        succeed, task = self.master_client.get_task("test-ds")
        self.assertTrue(succeed)
        self.assertEqual(task.shard.start, 0)
        self.assertEqual(task.shard.end, 640)
        succeed = self.master_client.report_task_result("test-ds", 0, "")
        self.assertTrue(succeed)

    def test_rdzv_manager(self):
        self.master_client.report_rdzv_params(1, 1, 360, 1, 600)
        self.master_client.join_rendezvous(
            0, 8, RendezvousName.ELASTIC_TRAINING
        )
        round, group, world = self.master_client.get_comm_world(
            RendezvousName.ELASTIC_TRAINING, 0
        )
        self.assertEqual(round, 1)
        self.assertEqual(group, 0)
        self.assertEqual(world[0], 8)

    def test_start_and_stop(self):
        mock_k8s_client()
        params = MockK8sPSJobArgs()
        params.initilize()
        master = DistributedJobMaster(2222, params)

        try:
            master.prepare()
            active_threads_name = [t.name for t in threading.enumerate()]
            self.assertIn("job_diagnosing", active_threads_name)
            master.request_stop(True, "", "")
        except Exception:
            pass

    def test_pre_check(self):
        self.assertEqual(
            self.job_context.get_pre_check_status(), PreCheckStatus.CHECKING
        )
        self._master.pre_check()
        self.assertEqual(
            self.job_context.get_pre_check_status(), PreCheckStatus.DISABLED
        )
