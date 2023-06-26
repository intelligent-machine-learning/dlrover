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

import os
import pickle
from concurrent import futures

import grpc
from google.protobuf import empty_pb2

from atorch.auto.engine.task import TaskType
from atorch.common.log_utils import default_logger as logger
from atorch.protos import acceleration_pb2, acceleration_pb2_grpc


def create_acceleration_service(port, executor, pool_size=None):
    if pool_size is None:
        pool_size = int(os.getenv("WORLD_SIZE", 8))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=pool_size),
    )
    acceleration_service = AutoAccelerationService(executor)
    acceleration_pb2_grpc.add_AutoAccelerationServiceServicer_to_server(acceleration_service, server)
    server.add_insecure_port("[::]:{}".format(port))
    logger.info("AutoAccelerationService has been created with port {}.".format(port))
    return server


class AutoAccelerationService(acceleration_pb2_grpc.AutoAccelerationServiceServicer):
    def __init__(self, executor):
        self._executor = executor

    def get_task(self, get_task_request, _):
        process_id = get_task_request.process_id
        gotten_task = self._executor.get_task(process_id)
        task = acceleration_pb2.AutoAccelerationTask(
            task_id=gotten_task.task_id,
            task_type=gotten_task.task_type,
            process_mode=gotten_task.process_mode,
            time_limit=gotten_task.time_limit,
        )
        if gotten_task.task_type == TaskType.ANALYSE:
            for names in gotten_task.task_info:
                task.analysis_method.names.append(names)
        elif (
            gotten_task.task_type in [TaskType.TUNE, TaskType.DRYRUN, TaskType.FINISH]
            and gotten_task.task_info is not None
        ):
            for info in gotten_task.task_info:
                opt_method = acceleration_pb2.OptimizationMethod(name=info[0], config=info[1], tunable=info[2])
                task.strategy.opt.append(opt_method)
        elif gotten_task.task_type == TaskType.SETUP_PARALLEL_GROUP:
            task.parallel_group_info = pickle.dumps(gotten_task.task_info)
        return task

    def report_task_result(self, request, _):
        task_id = request.task_id
        process_id = request.process_id
        status = request.status
        task_type = request.task_type
        result = None
        if task_type == TaskType.DRYRUN:
            result = pickle.loads(request.dryrun_result)
        elif task_type == TaskType.TUNE:
            result = []
            for opt_method in request.strategy.opt:
                result.append(
                    (
                        opt_method.name,
                        opt_method.config,
                        opt_method.tunable,
                    )
                )
        elif task_type == TaskType.ANALYSE:
            result = pickle.loads(request.model_meta)
        elif task_type == TaskType.FINISH:
            result = None
        self._executor.report_task_result(task_id, process_id, status, result)
        return empty_pb2.Empty()
