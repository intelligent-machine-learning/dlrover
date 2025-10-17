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


MASTER_STATE_KEY_PREFIX = (
    "DLROVER_MASTER_STATE:"  # f"DLROVER_MASTER_STATE:{config.job_name}"
)
RAY_HANG_CHECK_INTERVAL = 5.0  # Interval for monitoring actor invocations
RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME = 60
JOB_OPTIONS_ENV_PREFIX = "DLROVER_UNIFIED_"


class InternalDLWorkloadRole(object):
    ELASTIC_ROLE = "ELASTIC"
    GLOBAL_ROLE = "GLOBAL"


class DLWorkloadEnv(object):
    JOB = "JOB"
    NAME = "NAME"
    ROLE = "ROLE"
    RANK = "RANK"
    WORLD_SIZE = "WORLD_SIZE"
    LOCAL_RANK = "LOCAL_RANK"
    LOCAL_WORLD_SIZE = "LOCAL_WORLD_SIZE"
    MASTER_ADDR = "MASTER_ADDR"
    MASTER_PORT = "MASTER_PORT"

    WORKING_DIR = "DLROVER_WORKING_DIR"

    DEVICE_COLLOCATION_GROUP = "DEVICE_COLLOCATION_GROUP"

    RAY_NOSET_VISIBLE_DEVICES_ENVS = {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "true",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "true",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "true",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES": "true",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES": "true",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "true",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR": "true",
    }
