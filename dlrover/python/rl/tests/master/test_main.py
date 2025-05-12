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

import ray

from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.enums import JobStage
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.main import DLRoverRLMaster
from dlrover.python.rl.tests.test_data import TestData


class RLMasterTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=1, ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_master_actor(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_0}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        job_config = JobConfig.build_from_args(parse_job_args(args))

        master = DLRoverRLMaster.remote(
            job_config.serialize(), rl_context.serialize()
        )
        self.assertIsNotNone(master)
        self.assertEqual(ray.get(master.get_job_stage.remote()), JobStage.INIT)
        self.assertFalse(ray.get(master.is_job_started.remote()))
