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
import argparse

from omegaconf import DictConfig

from dlrover.python.unified.common.args import (
    _parse_master_state_backend_type,
    _parse_omega_config,
    _parse_scheduling_strategy_type,
    parse_job_args,
)
from dlrover.python.unified.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)
from dlrover.python.unified.tests.base import BaseTest


class ArgsTest(BaseTest):
    def test_parse_master_state_backend_type(self):
        self.assertEqual(
            _parse_master_state_backend_type("ray_internal"),
            MasterStateBackendType.RAY_INTERNAL,
        )
        self.assertEqual(
            _parse_master_state_backend_type("HDFS"),
            MasterStateBackendType.HDFS,
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_master_state_backend_type("local")

    def test_parse_scheduling_strategy_type(self):
        self.assertEqual(
            _parse_scheduling_strategy_type("simple"),
            SchedulingStrategyType.SIMPLE,
        )

        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_scheduling_strategy_type("test")

    def test_parse_trainer_config(self):
        self.assertIsNotNone(_parse_omega_config({}))
        self.assertIsNotNone(_parse_omega_config("{}"))
        self.assertEqual(_parse_omega_config('{"k1": "v1"}').get("k1"), "v1")
        self.assertEqual(_parse_omega_config({"k1": "v1"}).get("k1"), "v1")
        self.assertEqual(
            _parse_omega_config({"k1": "v1", "k2": {"k22": "v2"}})
            .get("k2")
            .get("k22"),
            "v2",
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_omega_config(123)

    def test_parsing(self):
        args = ["--job_name", "test", "--dl_type", "SFT", "--dl_config", "{}"]
        parsed_args = parse_job_args(args)
        self.assertEqual(parsed_args.job_name, "test")
        self.assertEqual(parsed_args.master_cpu, 2)
        self.assertEqual(parsed_args.master_mem, 4096)
        self.assertEqual(
            parsed_args.master_state_backend_type,
            MasterStateBackendType.RAY_INTERNAL,
        )
        self.assertEqual(parsed_args.master_state_backend_config, {})
        self.assertEqual(
            parsed_args.scheduling_strategy_type, SchedulingStrategyType.AUTO
        )
        self.assertEqual(parsed_args.job_max_restart, 10)
        self.assertEqual(parsed_args.master_max_restart, 10)
        self.assertEqual(parsed_args.workload_max_restart, {})

        args = [
            "--job_name",
            "test",
            "--master_cpu",
            "4",
            "--master_mem",
            "8192",
            "--master_state_backend_type",
            "hdfs",
            "--master_state_backend_config",
            '{"path": "test_path"}',
            "--max_restart",
            "11",
            "--workload_max_restart",
            "{'actor': 10}",
            "--dl_type",
            "SFT",
            "--dl_config",
            "{}",
        ]
        parsed_args = parse_job_args(args)
        self.assertEqual(parsed_args.master_cpu, 4)
        self.assertEqual(parsed_args.master_mem, 8192)
        self.assertEqual(
            parsed_args.master_state_backend_type, MasterStateBackendType.HDFS
        )
        self.assertEqual(
            parsed_args.master_state_backend_config, {"path": "test_path"}
        )
        self.assertEqual(parsed_args.job_max_restart, 11)
        self.assertEqual(parsed_args.workload_max_restart, {"actor": 10})

        args = [
            "--job_name",
            "test",
            "--master_cpu",
            "4",
            "--master_memory",
            "8192",
            "--master_state_backend_type",
            "hdfs",
            "--master_state_backend_config",
            '{"path": "test_path"}',
            "--dl_type",
            "RL",
            "--dl_config",
            '{"algorithm_type":"GRPO","config":{"c1":"v1"},"trainer":{"type":'
            '"USER_DEFINED","module":"dlrover.python.rl.tests.test_class",'
            '"class":"TestInteractiveTrainer"},"workload":{"actor":{"num":2,'
            '"module":"dlrover.python.rl.tests.test_class","class":'
            '"TestActor","resource":{"cpu":0.1}},"rollout":{"num":1,"module":'
            '"dlrover.python.rl.tests.test_class","class":"TestRollout",'
            '"resource":{"cpu":0.1}},"reference":{"num":2,"module":'
            '"dlrover.python.rl.tests.test_class","class":"TestReference",'
            '"resource":{"cpu":0.1}},"reward":{"num":1,"module":'
            '"dlrover.python.rl.tests.test_class","class":"TestReward",'
            '"resource":{"cpu":0.1}}}}',
        ]
        parsed_args = parse_job_args(args)
        self.assertEqual(parsed_args.master_cpu, 4)
        self.assertEqual(parsed_args.master_mem, 8192)
        rl_config: DictConfig = parsed_args.dl_config
        self.assertIsNotNone(rl_config)
        self.assertEqual(rl_config.get("algorithm_type"), "GRPO")
        self.assertEqual(rl_config.get("config").get("c1"), "v1")
