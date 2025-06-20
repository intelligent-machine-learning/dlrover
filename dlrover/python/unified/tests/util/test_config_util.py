#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from omegaconf import DictConfig, OmegaConf

from dlrover.python.unified.tests.base import BaseTest
from dlrover.python.unified.util.config_util import (
    args_2_omega_conf,
    convert_str_values,
    omega_conf_2_args,
)


class ConfigUtilTest(BaseTest):
    def test_args_2_omega_transfer(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--test0", type=str, default=None)
        parser.add_argument("--test1", type=int, default=0)
        parser.add_argument("--test2", type=float, default=1.1)

        args = []
        parsed_args = parser.parse_args(args)
        self.assertTrue(isinstance(parsed_args, argparse.Namespace))
        self.assertEqual(parsed_args.test0, None)
        self.assertEqual(parsed_args.test1, 0)
        self.assertEqual(parsed_args.test2, 1.1)

        omegaconf = args_2_omega_conf(parsed_args)
        self.assertTrue(isinstance(omegaconf, DictConfig))
        self.assertEqual(omegaconf["test0"], parsed_args.test0)
        self.assertEqual(omegaconf["test1"], parsed_args.test1)
        self.assertEqual(omegaconf["test2"], parsed_args.test2)

        parsed_args0 = omega_conf_2_args(omegaconf)
        self.assertTrue(isinstance(parsed_args0, argparse.Namespace))
        self.assertEqual(parsed_args0.test0, parsed_args.test0)
        self.assertEqual(parsed_args0.test1, parsed_args.test1)
        self.assertEqual(parsed_args0.test2, parsed_args.test2)

    def test_convert_str_values(self):
        config = OmegaConf.create(
            {
                "k1": "None",
                "k2": "123",
                "k3": "45.67",
                "k4": "abc",
                "k5": {
                    "nested_k1": "None",
                    "nested_k2": "789",
                    "nested_k3": "12.34",
                },
                "k6": ["None", "567", "89.01", "text"],
            }
        )

        convert_str_values(config)
        self.assertIsNone(config["k1"])
        self.assertEqual(config["k3"], 45.67)
        self.assertIsNone(config["k5"]["nested_k1"])
        self.assertEqual(config["k5"]["nested_k2"], 789)
        self.assertIsNone(config["k6"][0])
        self.assertEqual(config["k6"][2], 89.01)
