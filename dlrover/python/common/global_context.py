# Copyright 2022 The DLRover Authors. All rights reserved.
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
import threading

from dlrover.python.common.constants import UserEnv
from dlrover.python.common.grpc import find_free_port_in_range
from dlrover.python.common.log import default_logger as logger


class ConfigKeys(object):
    TRAIN_SPEED_RECORD_NUM = "train_speed_record_num"
    SECONDS_TO_START_AUTOSCALE_WORKER = "seconds_to_start_autoscale_worker"
    STEP_TO_ADJUST_WORKER = "step_to_adjust_worker"
    OPTIMIZE_WORKER_CPU_THRESHOLD = "optimize_worker_cpu_threshold"
    SECONDS_FOR_STABLE_WORKER_COUNT = "seconds_for_stable_worker_count"
    SECONDS_INTERVAL_TO_OPTIMIZE = "seconds_interval_to_optimize"
    FACTOR_TO_CUT_PENDING_CPU = "factor_to_cut_pending_cpu"
    FACTOR_TO_CUT_PENDING_MEM = "factor_to_cut_pending_mem"
    SECONDS_TO_WAIT_PENDING_POD = "seconds_to_wait_pending_pod"
    SECONDS_HUGE_TRAINING_THRESHOLD = "seconds_huge_training_threshold"
    GLOBAL_STEP_COUNT_TO_AUTO_WORKER = "global_step_count_to_auto_worker"
    SECONDS_TO_CHANGE_PS = "seconds_to_change_ps"
    SECONDS_TO_WAIT_FAILED_PS = "seconds_to_wait_failed_ps"


class DefaultConfigValues(object):
    TRAIN_SPEED_RECORD_NUM = 50
    SECENDS_TO_START_AUTOSCALE_WORKER = 90
    STEP_TO_ADJUST_WORKER = 200
    OPTIMIZED_WORKER_CPU_THRESHOLD = 20
    SECONDS_FOR_STABLE_WORKER_COUNT = 60
    SECONDS_INTERVAL_TO_OPTIMIZE = 300
    DFACTOR_TO_CUT_PENDING_CPU = 2
    FACTOR_TO_CUT_PENDING_MEM = 2
    SECONDS_TO_WAIT_PENDING_POD = 900  # 15min
    SECONDS_HUGE_TRAINING_THRESHOLD = 1800  # 30min
    GLOBAL_STEP_COUNT_TO_AUTO_WORKER = 5
    SECONDS_TO_CHANGE_PS = 3600  # 1h
    SECONDS_TO_WAIT_FAILED_PS = 600  # 10min


class Context(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self.train_speed_record_num = 0
        self.seconds_to_autoscale_worker = 0
        self.step_to_adjust_worker = 0
        self.optimize_worker_cpu_threshold = 0
        self.seconds_for_stable_worker_count = 0
        self.seconds_interval_to_optimize = 0
        self.factor_to_cut_pending_cpu = 0
        self.factor_to_cut_pending_mem = 0
        self.seconds_to_wait_pending_pod = 0
        self.seconds_huge_training_threshold = 0
        self.sample_count_to_adjust_worker = 0
        self.seconds_interval_to_change_ps = 0
        self.seconds_to_wait_failed_ps = 0
        self.auto_worker_enabled = False
        self.auto_ps_enabled = False
        self.is_tfv1_ps = False
        self.master_port = 0
        self.relaunch_error = False
        self.print_config()

    def set_params_from_brain(self):
        self.train_speed_record_num = self.get_param_value_from_brain(
            ConfigKeys.TRAIN_SPEED_RECORD_NUM,
            DefaultConfigValues.TRAIN_SPEED_RECORD_NUM,
        )
        self.seconds_to_autoscale_worker = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_START_AUTOSCALE_WORKER,
            DefaultConfigValues.SECENDS_TO_START_AUTOSCALE_WORKER,
        )
        self.step_to_adjust_worker = self.get_param_value_from_brain(
            ConfigKeys.STEP_TO_ADJUST_WORKER,
            DefaultConfigValues.STEP_TO_ADJUST_WORKER,
        )
        self.optimize_worker_cpu_threshold = self.get_param_value_from_brain(
            ConfigKeys.OPTIMIZE_WORKER_CPU_THRESHOLD,
            DefaultConfigValues.OPTIMIZED_WORKER_CPU_THRESHOLD,
        )
        self.seconds_for_stable_worker_count = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_FOR_STABLE_WORKER_COUNT,
            DefaultConfigValues.SECONDS_FOR_STABLE_WORKER_COUNT,
        )
        self.seconds_interval_to_optimize = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_INTERVAL_TO_OPTIMIZE,
            DefaultConfigValues.SECONDS_INTERVAL_TO_OPTIMIZE,
        )
        self.factor_to_cut_pending_cpu = self.get_param_value_from_brain(
            ConfigKeys.FACTOR_TO_CUT_PENDING_CPU,
            DefaultConfigValues.DFACTOR_TO_CUT_PENDING_CPU,
        )
        self.factor_to_cut_pending_mem = self.get_param_value_from_brain(
            ConfigKeys.FACTOR_TO_CUT_PENDING_MEM,
            DefaultConfigValues.FACTOR_TO_CUT_PENDING_MEM,
        )
        self.seconds_to_wait_pending_pod = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_WAIT_PENDING_POD,
            DefaultConfigValues.SECONDS_TO_WAIT_PENDING_POD,
        )
        self.seconds_huge_training_threshold = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_HUGE_TRAINING_THRESHOLD,
            DefaultConfigValues.SECONDS_HUGE_TRAINING_THRESHOLD,
        )
        self.sample_count_to_adjust_worker = self.get_param_value_from_brain(
            ConfigKeys.GLOBAL_STEP_COUNT_TO_AUTO_WORKER,
            DefaultConfigValues.GLOBAL_STEP_COUNT_TO_AUTO_WORKER,
        )
        self.seconds_interval_to_change_ps = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_CHANGE_PS,
            DefaultConfigValues.SECONDS_TO_CHANGE_PS,
        )
        self.seconds_to_wait_failed_ps = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_WAIT_FAILED_PS,
            DefaultConfigValues.SECONDS_TO_WAIT_FAILED_PS,
        )

    def config_master_port(self, port=0):
        if port > 0:
            self.master_port = port
        else:
            self.master_port = find_free_port_in_range(50001, 65535)

    def get_param_value_from_brain(self, key_name, default_value, dtype=int):
        """TODO: Get the configured value from Brain service."""
        value = default_value
        return dtype(value)

    def print_config(self):
        logger.info("DLRover global context = {}".format(self.__dict__))

    @property
    def user_id(self):
        return os.getenv(UserEnv.USER_ID, "")

    @classmethod
    def singleton_instance(cls, *args, **kwargs):
        if not hasattr(Context, "_instance"):
            with Context._instance_lock:
                if not hasattr(Context, "_instance"):
                    Context._instance = Context(*args, **kwargs)
        return Context._instance
