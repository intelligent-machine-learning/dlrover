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


class TaskType:
    ANALYSE = "ANALYSE"
    TUNE = "TUNE"
    FINISH = "FINISH"
    SETUP_PARALLEL_GROUP = "SETUP_PARALLEL_GROUP"
    DRYRUN = "DRYRUN"
    WAIT = "WAIT"
    FAIL = "FAIL"


class TaskProcessMode:
    """
    A task may be run on one or more processes.
    For example, an ANALYSE task may use ONE_PROCESS, while a DRYRUN,
    a SETUP_PARALLEL_GROUP task or a FINISH task will use ALL_PROCESS.
    A TUNE task may use ONE_PROCESS, or ALL_PROCESS.
    TODO: support custom parallel group
    """

    ONE_PROCESS = "ONE_PROCESS"
    ALL_PROCESS = "ALL_PROCESS"


class TaskStatus:
    PENDING = 0
    ASSIGNING = 1
    RUNNING = 2
    CANCELLED = 3
    FAILED = 4
    SUCCEEDED = 5


class Task(object):
    """
    This Task definition is a superset of Task in atorch.auto.task.
    This Task also includes status, process mode, task result.
    Task status change:
        "ONE_PROCESS": PENDING -> RUNNING -> SUCCEEDED/FAILED
        "ALL_PROCESS": PENDING -> ASSIGNING-> RUNNING -> SUCCEEDED/FAILED
    """

    def __init__(
        self,
        task_type,
        task_info,
        task_id=-1,
        strategy_id=-1,
        process_mode=TaskProcessMode.ONE_PROCESS,
        time_limit=None,
        task_status=TaskStatus.PENDING,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.task_info = task_info
        self.time_limit = time_limit
        self.process_mode = process_mode
        self.process_assigned = []  # list of process id
        self.status = task_status
        self.task_result = None
        self.strategy_id = strategy_id

    def add_process(self, process_id):
        self.process_assigned.append(process_id)
