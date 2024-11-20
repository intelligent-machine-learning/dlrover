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

import unittest

from dlrover.python.master.args import parse_master_args
from dlrover.trainer.torch.elastic_run import parse_args


class ArgsTest(unittest.TestCase):
    def test_parse_master_args(self):
        original_args = [
            "--job_name",
            "test",
            "--namespace",
            "default",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.job_name, "test")
        self.assertTrue(parsed_args.namespace, "default")

    def test_parse_agent_args(self):
        original_args = [
            "--training_log_file",
            "/tmp/dlrover",
            "--step_rank",
            "-1",
            "--step_pattern",
            r"""^\s+\[(\d{4}\-\d{2}\-\d{2} \d{2}:\d{2}:\d{2})\]
            \s+iteration\s+(\d+)\/\s+(\d+)\s+\|""",
            "sleep",
            "60",
        ]
        parsed_args = parse_args(original_args)
        self.assertEqual(parsed_args.training_log_file, "/tmp/dlrover")
        self.assertEqual(parsed_args.step_rank, -1)
        self.assertEqual(
            parsed_args.step_pattern,
            r"""^\s+\[(\d{4}\-\d{2}\-\d{2} \d{2}:\d{2}:\d{2})\]
            \s+iteration\s+(\d+)\/\s+(\d+)\s+\|""",
        )
