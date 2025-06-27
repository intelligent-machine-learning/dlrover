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

import ray

from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.dl_context import DLContext, RLContext
from dlrover.python.unified.common.enums import JobStage
from dlrover.python.unified.master.elastic.master import ElasticMaster
from dlrover.python.unified.master.mpmd.master import MPMDMaster
from dlrover.python.unified.tests.base import RayBaseTest
from dlrover.python.unified.tests.test_data import TestData


class DLMasterTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        self.init_ray_safely(num_cpus=1)

    def tearDown(self):
        self.close_ray_safely()
        super().tearDown()

    def test_master_actor(self):
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_0}",
        ]
        rl_context = RLContext.build_from_args(parse_job_args(args))
        job_config = JobConfig.build_from_args(parse_job_args(args))

        master = MPMDMaster.remote(
            job_config.serialize(), rl_context.serialize()
        )
        self.assertIsNotNone(master)
        self.assertEqual(ray.get(master.get_job_stage.remote()), JobStage.INIT)
        self.assertFalse(ray.get(master.is_job_started.remote()))

        args = [
            "--job_name",
            "test",
            "--dl_type",
            "SFT",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_SFT_CONF_0}",
        ]
        dl_context = DLContext.build_from_args(parse_job_args(args))
        job_config = JobConfig.build_from_args(parse_job_args(args))

        master = ElasticMaster.remote(
            job_config.serialize(), dl_context.serialize()
        )
        self.assertIsNotNone(master)
        self.assertEqual(ray.get(master.get_job_stage.remote()), JobStage.INIT)
        self.assertFalse(ray.get(master.is_job_started.remote()))
