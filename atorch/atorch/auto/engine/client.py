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

import grpc

from atorch.protos import acceleration_pb2, acceleration_pb2_grpc

GRPC_MAX_SEND_MESSAGE_LENGTH = 256 * 1024 * 1024
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 256 * 1024 * 1024


def build_channel(addr):
    channel = grpc.insecure_channel(
        addr,
        options=[
            ("grpc.max_send_message_length", GRPC_MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC_MAX_RECEIVE_MESSAGE_LENGTH,
            ),
            ("grpc.enable_retries", True),
            (
                "grpc.service_config",
                """{ "retryPolicy":{ "maxAttempts": 5, "initialBackoff": "0.2s", \n
"maxBackoff": "3s", "backoffMutiplier": 2, \n
"retryableStatusCodes": [ "UNAVAILABLE" ] } }""",
            ),
        ],
    )
    return channel


class AutoAccelerationClient(object):
    def __init__(self, channel, process_id):
        self._channel = channel
        self._stub = acceleration_pb2_grpc.AutoAccelerationServiceStub(self._channel)
        self._process_id = process_id

    def __del__(self):
        self._channel.close()

    def get_task(self):
        request = acceleration_pb2.GetAutoAccelerationTaskRequest(process_id=self._process_id)
        res = self._stub.get_task(request)
        task_id = res.task_id
        task_type = res.task_type
        process_mode = res.process_mode
        time_limit = res.time_limit
        task_info = None
        if task_type in ["TUNE", "DRYRUN", "FINISH"]:
            task_info = []
            for opt in res.strategy.opt:
                task_info.append((opt.name, opt.config, opt.tunable))
        elif task_type == "ANALYSE":
            task_info = list(res.analysis_method.names)
        elif task_type == "SETUP_PARALLEL_GROUP":
            task_info = res.parallel_group_info
        return task_id, task_type, process_mode, time_limit, task_info

    def report_task_result(
        self,
        task_id,
        task_type,
        status,
        result,
    ):
        task_result = acceleration_pb2.AutoAccelerationTaskResult()
        task_result.task_id = task_id
        task_result.process_id = self._process_id
        task_result.status = status
        task_result.task_type = task_type
        if task_type == "DRYRUN":
            task_result.dryrun_result = result
        elif task_type == "TUNE":
            for opt in result:
                method = acceleration_pb2.OptimizationMethod(name=opt[0], config=opt[1], tunable=opt[2])
                task_result.strategy.opt.append(method)
        elif task_type == "ANALYSE":
            task_result.model_meta = result
        self._stub.report_task_result(task_result)


def build_auto_acc_client(addr=None, process_id=None):
    if addr is None:
        addr = os.getenv("MASTER_ADDR", "")
    if process_id is None:
        process_id = int(os.getenv("RANK", "0"))
    channel = build_channel(addr)
    auto_acc_client = AutoAccelerationClient(channel, process_id)
    return auto_acc_client


class GlobalAutoAccelerationClient(object):
    AUTO_ACC_CLIENT = build_auto_acc_client()
