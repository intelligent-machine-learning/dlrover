# Copyright 2023 The DLRover Authors. All rights reserved.
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

import json
import os

from dlrover.trainer.util.log_util import default_logger as logger


def get_tf_config():
    tf_config = json.loads(os.environ.get("TF_CONFIG") or "{}")
    if not tf_config:
        logger.error(
            "TF_CONFIG should not be empty in distributed environment."
        )
        raise Exception(
            "TF_CONFIG should not be empty in distributed environment."
        )
    return tf_config


def get_tf_config_task_type_and_index():
    tf_config = get_tf_config()
    task = tf_config.get("task", None)
    if task is None:
        raise Exception(
            "TF_CONFIG task should not be empty \
                in distributed environment."
        )
    task_type = task.get("type", None)
    task_index = task.get("index", None)
    if task_type is None or task_index is None:
        raise Exception(
            "TF_CONFIG task type or index should\
              not be empty in distributed environment."
        )
    return task_type, task_index
