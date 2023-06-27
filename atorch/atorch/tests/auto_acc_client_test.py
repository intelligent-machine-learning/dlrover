# Copyright 2022 The ElasticDL Authors. All rights reserved.
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

import pickle
import unittest
from unittest import mock

from atorch.auto.engine.client import GlobalAutoAccelerationClient
from atorch.protos import acceleration_pb2


class AutoAccelerationClient(unittest.TestCase):
    def setUp(self):
        self.client = GlobalAutoAccelerationClient.AUTO_ACC_CLIENT

    def test_get_analyse_task(self):
        method = acceleration_pb2.AnalysisMethod()
        method.names.extend(["default", "self-defined"])
        task = acceleration_pb2.AutoAccelerationTask(
            task_id=0,
            task_type="ANALYSE",
            process_mode="ONE_PROCESS",
            analysis_method=method,
            time_limit=30,
        )
        self.client._stub.get_task = mock.MagicMock(return_value=task)
        (
            task_id,
            task_type,
            process_mode,
            time_limit,
            task_info,
        ) = self.client.get_task()
        self.assertEqual(task_id, 0)
        self.assertEqual(task_type, "ANALYSE")
        self.assertEqual(process_mode, "ONE_PROCESS")
        self.assertEqual(time_limit, 30)
        self.assertEqual(task_info, ["default", "self-defined"])

    def test_get_parallel_task(self):
        parallel_group_info = {
            "model_parallel_size": 3,
            "model_parallel_group": [[0, 1, 2], [3, 4, 5]],
        }
        task = acceleration_pb2.AutoAccelerationTask(
            task_id=1,
            task_type="SETUP_PARALLEL_GROUP",
            process_mode="ALL_PROCESS",
            parallel_group_info=pickle.dumps(parallel_group_info),
            time_limit=5,
        )
        self.client._stub.get_task = mock.MagicMock(return_value=task)
        (
            task_id,
            task_type,
            process_mode,
            time_limit,
            task_info,
        ) = self.client.get_task()
        self.assertEqual(task_id, 1)
        self.assertEqual(task_type, "SETUP_PARALLEL_GROUP")
        self.assertEqual(process_mode, "ALL_PROCESS")
        self.assertEqual(time_limit, 5)
        self.assertEqual(parallel_group_info, pickle.loads(task_info))

    def test_get_strategy_task(self):
        parallel_group_info = {
            "model_parallel_size": 3,
            "model_parallel_group": [[0, 1, 2], [3, 4, 5]],
        }
        methods = [
            ("1D", parallel_group_info, True),
            ("2D", parallel_group_info, False),
        ]
        opt_method0 = acceleration_pb2.OptimizationMethod(
            name=methods[0][0],
            config=pickle.dumps(methods[0][1]),
            tunable=methods[0][2],
        )
        opt_method1 = acceleration_pb2.OptimizationMethod(
            name=methods[1][0],
            config=pickle.dumps(methods[1][1]),
            tunable=methods[1][2],
        )
        strategy = acceleration_pb2.Strategy()
        strategy.opt.extend([opt_method0, opt_method1])
        task = acceleration_pb2.AutoAccelerationTask(
            task_id=2,
            task_type="TUNE",
            process_mode="ONE_MODEL_PARALLEL_GROUP",
            strategy=strategy,
            time_limit=600,
        )
        self.client._stub.get_task = mock.MagicMock(return_value=task)
        (
            task_id,
            task_type,
            process_mode,
            time_limit,
            task_info,
        ) = self.client.get_task()
        self.assertEqual(task_id, 2)
        self.assertEqual(task_type, "TUNE")
        self.assertEqual(process_mode, "ONE_MODEL_PARALLEL_GROUP")
        self.assertEqual(time_limit, 600)
        for method, info in zip(methods, task_info):
            self.assertEqual(method[0], info[0])
            self.assertEqual(method[1], pickle.loads(info[1]))
            self.assertEqual(method[2], info[2])

    def test_report_more_than_one_result(self):
        self.client._stub.report_task_result = mock.MagicMock(return_value=None)
        self.client.report_task_result(0, "TUNE", True, [("method", pickle.dumps({}), False)])
        self.client.report_task_result(0, "DRYRUN", True, pickle.dumps({}))
        self.client.report_task_result(0, "ANALYSE", True, pickle.dumps({}))
