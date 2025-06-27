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
import os

import ray

from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.driver.main import get_master_cls, main
from dlrover.python.unified.master.elastic.master import ElasticMaster
from dlrover.python.unified.master.mpmd.master import MPMDMaster
from dlrover.python.unified.tests.base import BaseTest, RayBaseTest
from dlrover.python.unified.tests.test_data import TestData


class DriverTest(BaseTest):
    def test_get_master_cls(self):
        self.assertEqual(
            get_master_cls(
                parse_job_args(
                    [
                        "--job_name",
                        "test",
                        "--dl_type",
                        "RL",
                        "--dl_config",
                        "{}",
                    ]
                )
            ),
            MPMDMaster,
        )
        self.assertEqual(
            get_master_cls(
                parse_job_args(
                    [
                        "--job_name",
                        "test",
                        "--dl_type",
                        "SFT",
                        "--dl_config",
                        "{}",
                    ]
                )
            ),
            ElasticMaster,
        )


class DriverRayTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        os.environ[DLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"
        self.init_ray_safely()

    def tearDown(self):
        self.close_ray_safely()
        os.environ.clear()
        super().tearDown()

    def test_driver_all(self):
        args = [
            "--job_name",
            "test",
            "--master_cpu",
            "1",
            "--master_memory",
            "100",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_0}",
        ]

        main(args)
        self.assertIsNotNone(ray.get_actor("DLMaster-test"))

    def test_driver_all_with_ray_init(self):
        self.close_ray_safely()

        args = [
            "--job_name",
            "test",
            "--master_cpu",
            "1",
            "--master_memory",
            "100",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_RL_CONF_0}",
        ]

        main(args, ray_address="test")
        self.assertIsNotNone(ray.get_actor("DLMaster-test"))
