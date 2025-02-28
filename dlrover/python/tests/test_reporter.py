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

from master.elastic_training.net_topology import NodeTopologyMeta
from master.elastic_training.rdzv_manager import RendezvousParameters

from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.node import Node
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.training_event import DLRoverMasterEvent


class EventReporterTest(unittest.TestCase):
    def setUp(self):
        self.reporter = get_event_reporter()
        self.args = JobArgs("default", "test", "test")
        self.master_evt = DLRoverMasterEvent().singleton_instance()
        self.job_evt = self.master_evt.train_job(
            job_name=self.args.job_name, args=vars(self.args)
        )

    def test_inner_report(self):
        self.reporter.inner_report("1", "2", "3", "", {})

    def test_master_report(self):
        self.reporter.report_master_start(self.args)
        self.reporter.report_master_end(self.args, 0)
        self.reporter.report_job_start(self.job_evt, self.args)
        self.reporter.report_job_success(self.job_evt, self.args)
        self.reporter.report_job_fail(self.job_evt, self.args, "error")

    def test_manager_report(self):
        self.reporter.report_node_relaunch(
            Node("worker", 0), Node("worker", 1)
        )
        self.reporter.report_node_no_heartbeat(Node("worker", 0), 10)
        self.reporter.report_node_status_change(
            Node("worker", 0), "INIT", "PENDING"
        )

    def test_rdzv_report(self):
        rdzv_evt = self.master_evt.rendezvous(
            rendezvous_type="test",
            round_num=0,
            timeout_sec=1,
            max_nodes=1,
            min_nodes=1,
        )
        node_meta = NodeTopologyMeta()
        rdzv_params = RendezvousParameters(0, 0)
        self.reporter.report_rdzv_node_join(
            node_meta,
            "test",
            0,
            rdzv_params,
            waiting_nodes={0: node_meta},
            node_elapsed_time=1,
        )
        self.reporter.report_rdzv_timeout(
            rdzv_evt, "test", 0, rdzv_params, node_group=[], elapsed_time=1
        )
        self.reporter.report_rdzv_complete(
            rdzv_evt, "test", 0, rdzv_params, node_ids=[], elapsed_time=1
        )
