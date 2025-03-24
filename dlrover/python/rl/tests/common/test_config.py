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

from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.enums import MasterStateBackendType


class JobConfigTest(unittest.TestCase):
    def test_building(self):
        args = [
            "--job_name",
            "test",
            "--rl_config",
            "{}",
        ]
        job_config = JobConfig.build_from_args(parse_job_args(args))
        self.assertIsNotNone(job_config)
        self.assertEqual(
            job_config.master_state_backend_type,
            MasterStateBackendType.RAY_INTERNAL,
        )
        self.assertEqual(job_config.master_state_backend_config, {})

        args = [
            "--job_name",
            "test",
            "--master_state_backend_type",
            "hdfs",
            "--master_state_backend_config",
            "{'k1': 'v1'}",
            "--rl_config",
            "{}",
        ]
        job_config = JobConfig.build_from_args(parse_job_args(args))
        self.assertEqual(
            job_config.master_state_backend_type, MasterStateBackendType.HDFS
        )
        self.assertEqual(job_config.master_state_backend_config, {"k1": "v1"})

    def test_serialization(self):
        args = [
            "--job_name",
            "test",
            "--master_state_backend_type",
            "hdfs",
            "--master_state_backend_config",
            "{'k1': 'v1'}",
            "--rl_config",
            "{}",
        ]
        job_config = JobConfig.build_from_args(parse_job_args(args))
        serialized = job_config.serialize()
        self.assertIsNotNone(serialized)

        deserialized = JobConfig.deserialize(serialized)
        self.assertIsNotNone(deserialized)
        self.assertEqual(
            job_config.master_state_backend_type, MasterStateBackendType.HDFS
        )
        self.assertEqual(job_config.master_state_backend_config, {"k1": "v1"})
