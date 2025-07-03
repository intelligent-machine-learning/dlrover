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
import time

import ray

from dlrover.python.unified.api.rl import RLJobBuilder
from dlrover.python.unified.common.enums import JobStage
from dlrover.python.unified.master.mpmd.master import MPMDMaster
from dlrover.python.unified.tests.base import RayBaseTest

try:
    from ray.exceptions import ActorDiedError as ade
except ImportError:
    from builtins import RuntimeError as ade

from dlrover.python.unified.api.base import DLJobBuilder
from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.dl_context import RLContext
from dlrover.python.unified.tests.master.base import BaseMasterTest
from dlrover.python.unified.tests.test_data import TestData
from dlrover.python.util.function_util import timeout


class ApiFullTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        os.environ[DLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"
        self.init_ray_safely(num_cpus=8)

    def tearDown(self):
        self.close_ray_safely()
        super().tearDown()

    @timeout(20)
    def test_elastic_training(self):
        dl_job = (
            DLJobBuilder()
            .SFT_type()
            .node_num(3)
            .device_per_node(2)
            .device_type("CPU")
            .config({"c1": "v1"})
            .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
            .dlrover_run(
                "dlrover-run --nnodes=1:2 --nproc_per_node=2 "
                "--rdzv_conf join_timeout=600 --network_check "
                "--max-restarts=1 test.py",
                "dlrover.python.unified.tests.test_class",
                "TestElasticWorkload",
            )
            .build()
        )

        dl_job.submit("test", master_cpu=1, master_memory=128)

    @timeout(20)
    def test_elastic_training_with_error(self):
        dl_job = (
            DLJobBuilder()
            .SFT_type()
            .node_num(3)
            .device_per_node(2)
            .device_type("CPU")
            .config({"c1": "v1"})
            .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
            .dlrover_run(
                "dlrover-run --nnodes=1:2 --nproc_per_node=2 "
                "--rdzv_conf join_timeout=600 --network_check "
                "--max-restarts=1 test.py",
                "dlrover.python.unified.tests.test_class",
                "TestErrorElasticWorkload",
            )
            .build()
        )

        dl_job.submit(
            "test",
            master_cpu=1,
            master_memory=128,
            workload_max_restart={"ELASTIC": 1},
        )

    @timeout(20)
    def test_rl_0(self):
        rl_job = (
            RLJobBuilder()
            .node_num(3)
            .device_per_node(2)
            .device_type("CPU")
            .config({"c1": "v1"})
            .global_env({"e0": "v0"})
            .trainer(
                "dlrover.python.unified.tests.test_class",
                "TestInteractiveTrainer",
            )
            .actor("dlrover.python.unified.tests.test_class", "TestActor")
            .total(2)
            .per_node(1)
            .env({"e1": "v1"})
            .rollout("dlrover.python.unified.tests.test_class", "TestRollout")
            .total(2)
            .per_node(1)
            .reference(
                "dlrover.python.unified.tests.test_class", "TestReference"
            )
            .total(2)
            .per_node(1)
            .build()
        )

        rl_job.submit("test", master_cpu=1, master_memory=128)

    @timeout(20)
    def test_rl_1(self):
        rl_job = (
            RLJobBuilder()
            .node_num(1)
            .device_per_node(2)
            .device_type("CPU")
            .config({"c1": "v1"})
            .global_env({"e0": "v0"})
            .trainer(
                "dlrover.python.unified.tests.test_class",
                "TestInteractiveTrainer",
            )
            .actor("dlrover.python.unified.tests.test_class", "TestActor")
            .total(2)
            .per_node(2)
            .env({"e1": "v1"})
            .rollout("dlrover.python.unified.tests.test_class", "TestRollout")
            .total(2)
            .per_node(2)
            .reference(
                "dlrover.python.unified.tests.test_class", "TestReference"
            )
            .total(2)
            .per_node(2)
            .with_collocation("actor", "rollout", "reference")
            .build()
        )

        rl_job.submit("test", master_cpu=1, master_memory=128)


class RLMasterNormalTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_INTERACTIVE_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        rl_context = RLContext.build_from_args(parsed_args)
        self._job_context._dl_context = rl_context

        os.environ[DLMasterConstant.PG_STRATEGY_ENV] = "SPREAD"
        self.init_ray_safely(num_cpus=8)

    @timeout(20)
    def test(self):
        master_name = "test"

        master_actor = MPMDMaster.options(
            name=master_name,
            lifetime="detached",
        ).remote(
            self._job_context.job_config.serialize(),
            self._job_context.dl_context.serialize(),
        )

        ray.get(master_actor.ping.remote())
        master_actor.run.remote()
        time.sleep(3)

        # wait master done
        while True:
            try:
                result = ray.get(master_actor.get_job_status.remote())
            except ade:
                break
            if not JobStage.is_ending_stage(result):
                time.sleep(1)
            else:
                break


class RLMasterTrainerAbnormalTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        args = [
            "--job_name",
            "test",
            "--job_max_restart",
            "1",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_ERROR_TRAINER_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = RLContext.build_from_args(parsed_args)
        self._job_context._job_config = job_config
        self._job_context._dl_context = rl_context
        self.init_ray_safely(num_cpus=8)

    @timeout(30)
    def test_trainer_abnormal(self):
        master_name = "test"

        master_actor = MPMDMaster.options(
            name=master_name, lifetime="detached"
        ).remote(
            self._job_context.job_config.serialize(),
            self._job_context.dl_context.serialize(),
        )

        ray.get(master_actor.ping.remote())
        master_actor.run.remote()
        time.sleep(3)

        # wait master done
        while True:
            try:
                result = ray.get(master_actor.get_job_status.remote())
            except ade:
                break

            if not JobStage.is_ending_stage(result):
                time.sleep(1)
            else:
                break


class RLMasterTrainerWorkloadAbnormalTest(BaseMasterTest):
    def setUp(self):
        super().setUp()
        args = [
            "--job_name",
            "test",
            "--job_max_restart",
            "1",
            "--dl_type",
            "RL",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_WITH_ERROR_TRAINER_ACTOR_RL_CONF}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        rl_context = RLContext.build_from_args(parsed_args)
        self._job_context._job_config = job_config
        self._job_context._dl_context = rl_context
        self.init_ray_safely(num_cpus=8)

    @timeout(30)
    def test_trainer_workload_abnormal(self):
        master_name = "test"

        master_actor = MPMDMaster.options(
            name=master_name, lifetime="detached"
        ).remote(
            self._job_context.job_config.serialize(),
            self._job_context.dl_context.serialize(),
        )

        ray.get(master_actor.ping.remote())
        master_actor.run.remote()
        time.sleep(3)

        # wait master done
        while True:
            try:
                result = ray.get(master_actor.get_job_status.remote())
            except Exception:
                break
            if not JobStage.is_ending_stage(result):
                time.sleep(1)
            else:
                break
