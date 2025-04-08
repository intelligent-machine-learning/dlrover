# Copyright 2025 The EasyDL Authors. All rights reserved.
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


class RLMasterConstant(object):
    JOB_CONTEXT_STATE_KEY = "job-context"
    SCHEDULING_TIMEOUT_MIN_SECS = 30
    SCHEDULING_TIMEOUT_PER_ACTOR_SECS = 2
    RUN_WAIT_INTERVAL = 10
    EXIT_WAIT_INTERVAL = 10
    WORKLOAD_MAX_RESTART = 30


class RLJobStatus(object):
    INIT = "INIT"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class RLJobExitReason(object):
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class RLWorkloadEnv(object):
    NAME = "NAME"
    ROLE = "ROLE"
    RANK = "RANK"
    WORLD_SIZE = "WORLD_SIZE"
    LOCAL_RANK = "LOCAL_RANK"
    LOCAL_WORLD_SIZE = "LOCAL_WORLD_SIZE"
