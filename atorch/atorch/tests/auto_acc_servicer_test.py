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

from atorch.auto.engine.acceleration_engine import AccelerationEngine
from atorch.auto.engine.servicer import AutoAccelerationService
from atorch.auto.engine.task import Task, TaskProcessMode, TaskStatus, TaskType
from atorch.protos import acceleration_pb2


class AccelerationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        executor = AccelerationEngine.create_executor()
        self.servicer = AutoAccelerationService(executor)

    def test_get_strategy_task(self):
        config = {
            "pipeline_parallel_size": 3,
            "pipeline_parallel_group": [[0, 1, 2], [3, 4, 5]],
        }
        optimization_methods = [
            ("1F1B", pickle.dumps(config), True),
            ("bidirectional", pickle.dumps(config), False),
        ]
        task_id = 1
        task_type = TaskType.TUNE
        task_status = TaskStatus.PENDING
        process_mode = TaskProcessMode.ALL_PROCESS
        task = Task(
            task_type,
            optimization_methods,
            task_id=task_id,
            process_mode=process_mode,
            task_status=task_status,
            time_limit=5,
        )
        self.servicer._executor.get_task = mock.MagicMock(return_value=task)
        protobuf_task = self.servicer.get_task(acceleration_pb2.GetAutoAccelerationTaskRequest(), None)
        self.assertEqual(protobuf_task.task_id, task_id)
        self.assertEqual(protobuf_task.task_type, task_type)
        self.assertEqual(protobuf_task.process_mode, process_mode)
        self.assertEqual(protobuf_task.time_limit, 5)
        for method_tuple, method_proto in zip(optimization_methods, protobuf_task.strategy.opt):
            self.assertEqual(method_tuple[0], method_proto.name)
            self.assertEqual(method_tuple[1], method_proto.config)
            self.assertEqual(method_tuple[2], method_proto.tunable)

    def test_get_analysis_method_task(self):
        methods = ["default", "static", "dynamic"]
        task_id = 0
        task_type = TaskType.ANALYSE
        task_status = TaskStatus.PENDING
        process_mode = TaskProcessMode.ONE_PROCESS
        task = Task(
            task_type,
            methods,
            task_id=task_id,
            process_mode=process_mode,
            task_status=task_status,
            time_limit=30,
        )
        self.servicer._executor.get_task = mock.MagicMock(return_value=task)
        protobuf_task = self.servicer.get_task(acceleration_pb2.GetAutoAccelerationTaskRequest(), None)
        self.assertEqual(protobuf_task.task_id, task_id)
        self.assertEqual(protobuf_task.task_type, task_type)
        self.assertEqual(protobuf_task.process_mode, process_mode)
        self.assertEqual(protobuf_task.time_limit, 30)
        self.assertEqual(
            list(protobuf_task.analysis_method.names),
            methods,
        )

    def test_get_parallel_group_info_task(self):
        parallel_group_info = {
            "model_parallel_size": 3,
            "model_parallel_group": [[0, 1, 2], [3, 4, 5]],
        }

        task_id = 0
        task_type = TaskType.SETUP_PARALLEL_GROUP
        process_mode = TaskProcessMode.ALL_PROCESS
        task_status = TaskStatus.PENDING
        task = Task(
            task_type,
            parallel_group_info,
            task_id=task_id,
            process_mode=process_mode,
            task_status=task_status,
        )
        self.servicer._executor.get_task = mock.MagicMock(return_value=task)
        protobuf_task = self.servicer.get_task(acceleration_pb2.GetAutoAccelerationTaskRequest(), None)
        self.assertEqual(protobuf_task.task_id, task_id)
        self.assertEqual(protobuf_task.task_type, task_type)
        self.assertEqual(protobuf_task.process_mode, process_mode)
        self.assertEqual(protobuf_task.time_limit, 0)
        self.assertEqual(
            protobuf_task.parallel_group_info,
            pickle.dumps(parallel_group_info),
        )

    def test_report_strategy_result(self):
        self.servicer._executor.report_task_result = mock.MagicMock(return_value=None)
        task_result = acceleration_pb2.AutoAccelerationTaskResult(
            task_id=0,
            process_id=0,
            status=True,
            task_type="ANALYSE",
        )
        task_result.model_meta = pickle.dumps({"num_elements": 10})
        self.servicer.report_task_result(task_result, None)
