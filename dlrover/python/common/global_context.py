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
import importlib
import os

from dlrover.python.common.constants import (
    Accelerators,
    CommunicationType,
    UserEnv,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.util.common_util import (
    find_free_port_in_range,
    find_free_port_in_set,
)


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
    HANG_CPU_USAGE_RATE = "hang_cpu_usage_rate"


class DefaultValues(object):
    SERVICE_TYPE = CommunicationType.COMM_SERVICE_GRPC
    TRAIN_SPEED_RECORD_NUM = 50
    SEC_TO_START_AUTOSCALE_WORKER = 90
    STEP_TO_ADJUST_WORKER = 200
    OPTIMIZED_WORKER_CPU_THRESHOLD = 20
    SEC_FOR_STABLE_WORKER_COUNT = 60
    SEC_INTERVAL_TO_OPTIMIZE = 300
    FACTOR_TO_CUT_PENDING_CPU = 2
    FACTOR_TO_CUT_PENDING_MEM = 2
    SEC_TO_WAIT_PENDING_POD = 900  # 15min
    PENDING_FAIL_STRATEGY = 1  # 0: skip 1: necessary part 2: all
    SEC_HUGE_TRAINING_THRESHOLD = 1800  # 30min
    STEP_SAMPLE_COUNT_TO_AUTO_WORKER = 5
    SEC_TO_CHANGE_PS = 3600  # 1h
    SEC_TO_WAIT_FAILED_PS = 600  # 10min
    HANG_CPU_USAGE_RATE = 0.05
    HANG_DETECTION = 1
    HANG_DOWNTIME = 30
    MIN_HANG_DOWNTIME = 3
    GPU_NUM_PER_NODE = 8
    NPU_NUM_PER_NODE = 16
    MAX_METRIC_REC = 30
    PRE_CHECK_OPS = [
        (
            "dlrover.python.master.diagnosis.precheck_operator",
            "NoPreCheckOperator",
        )
    ]


class Context(Singleton):
    def __init__(self):
        self.master_service_type = DefaultValues.SERVICE_TYPE
        self.train_speed_record_num = DefaultValues.TRAIN_SPEED_RECORD_NUM
        self.seconds_to_autoscale_worker = (
            DefaultValues.SEC_TO_START_AUTOSCALE_WORKER
        )
        self.step_to_adjust_worker = DefaultValues.STEP_TO_ADJUST_WORKER
        self.optimize_worker_cpu_threshold = (
            DefaultValues.OPTIMIZED_WORKER_CPU_THRESHOLD
        )
        self.seconds_for_stable_worker_count = (
            DefaultValues.SEC_FOR_STABLE_WORKER_COUNT
        )
        self.seconds_interval_to_optimize = (
            DefaultValues.SEC_INTERVAL_TO_OPTIMIZE
        )
        self.factor_to_cut_pending_cpu = (
            DefaultValues.FACTOR_TO_CUT_PENDING_CPU
        )
        self.factor_to_cut_pending_mem = (
            DefaultValues.FACTOR_TO_CUT_PENDING_MEM
        )
        self.seconds_to_wait_pending_pod = (
            DefaultValues.SEC_TO_WAIT_PENDING_POD
        )
        self.pending_fail_strategy = DefaultValues.PENDING_FAIL_STRATEGY
        self.seconds_huge_training_threshold = (
            DefaultValues.SEC_HUGE_TRAINING_THRESHOLD
        )
        self.sample_count_to_adjust_worker = (
            DefaultValues.STEP_SAMPLE_COUNT_TO_AUTO_WORKER
        )
        self.hang_cpu_usage_percentage = DefaultValues.HANG_CPU_USAGE_RATE
        self.seconds_interval_to_change_ps = DefaultValues.SEC_TO_CHANGE_PS
        self.seconds_to_wait_failed_ps = DefaultValues.SEC_TO_WAIT_FAILED_PS
        self.auto_worker_enabled = False
        self.auto_ps_enabled = False
        self.is_tfv1_ps = False
        self.master_port = None
        self.relaunch_always = False
        # The strategy of 'hang detection':
        # 0: log only; 1: notify; 2: with fault tolerance
        self.hang_detection = DefaultValues.HANG_DETECTION
        # The duration of downtime as training hang, unit is minute
        self.hang_downtime = DefaultValues.HANG_DOWNTIME
        # The default xpu device type.
        self.xpu_type = Accelerators.NVIDIA_GPU
        self.pre_check_operators = DefaultValues.PRE_CHECK_OPS
        self.gpu_per_node = DefaultValues.GPU_NUM_PER_NODE
        self.npu_per_node = DefaultValues.NPU_NUM_PER_NODE

    def set_params_from_brain(self):
        self.train_speed_record_num = self.get_param_value_from_brain(
            ConfigKeys.TRAIN_SPEED_RECORD_NUM,
            DefaultValues.TRAIN_SPEED_RECORD_NUM,
        )
        self.seconds_to_autoscale_worker = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_START_AUTOSCALE_WORKER,
            DefaultValues.SEC_TO_START_AUTOSCALE_WORKER,
        )
        self.step_to_adjust_worker = self.get_param_value_from_brain(
            ConfigKeys.STEP_TO_ADJUST_WORKER,
            DefaultValues.STEP_TO_ADJUST_WORKER,
        )
        self.optimize_worker_cpu_threshold = self.get_param_value_from_brain(
            ConfigKeys.OPTIMIZE_WORKER_CPU_THRESHOLD,
            DefaultValues.OPTIMIZED_WORKER_CPU_THRESHOLD,
        )
        self.seconds_for_stable_worker_count = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_FOR_STABLE_WORKER_COUNT,
            DefaultValues.SEC_FOR_STABLE_WORKER_COUNT,
        )
        self.seconds_interval_to_optimize = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_INTERVAL_TO_OPTIMIZE,
            DefaultValues.SEC_INTERVAL_TO_OPTIMIZE,
        )
        self.factor_to_cut_pending_cpu = self.get_param_value_from_brain(
            ConfigKeys.FACTOR_TO_CUT_PENDING_CPU,
            DefaultValues.FACTOR_TO_CUT_PENDING_CPU,
        )
        self.factor_to_cut_pending_mem = self.get_param_value_from_brain(
            ConfigKeys.FACTOR_TO_CUT_PENDING_MEM,
            DefaultValues.FACTOR_TO_CUT_PENDING_MEM,
        )
        if self.seconds_to_wait_pending_pod > 0:
            self.seconds_to_wait_pending_pod = self.get_param_value_from_brain(
                ConfigKeys.SECONDS_TO_WAIT_PENDING_POD,
                DefaultValues.SEC_TO_WAIT_PENDING_POD,
            )
        self.seconds_huge_training_threshold = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_HUGE_TRAINING_THRESHOLD,
            DefaultValues.SEC_HUGE_TRAINING_THRESHOLD,
        )
        self.sample_count_to_adjust_worker = self.get_param_value_from_brain(
            ConfigKeys.GLOBAL_STEP_COUNT_TO_AUTO_WORKER,
            DefaultValues.STEP_SAMPLE_COUNT_TO_AUTO_WORKER,
        )
        self.seconds_interval_to_change_ps = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_CHANGE_PS,
            DefaultValues.SEC_TO_CHANGE_PS,
        )
        self.seconds_to_wait_failed_ps = self.get_param_value_from_brain(
            ConfigKeys.SECONDS_TO_WAIT_FAILED_PS,
            DefaultValues.SEC_TO_WAIT_FAILED_PS,
        )
        self.hang_cpu_usage_percentage = self.get_param_value_from_brain(
            ConfigKeys.HANG_CPU_USAGE_RATE,
            DefaultValues.HANG_CPU_USAGE_RATE,
        )

    def config_master_port(self, port=0):
        host_ports_env = os.getenv("HOST_PORTS", "")
        self.master_port = None
        if host_ports_env:
            ports = []
            for port in host_ports_env.split(","):
                ports.append(int(port))
            try:
                self.master_port = find_free_port_in_set(ports)
            except RuntimeError as e:
                logger.warning(e)
        elif port > 0:
            self.master_port = port
        if self.master_port is None:
            self.master_port = find_free_port_in_range(20000, 30000)

    def get_param_value_from_brain(self, key_name, default_value, dtype=float):
        """TODO: Get the configured value from Brain service."""
        value = default_value
        return dtype(value)

    def print_config(self):
        logger.info("DLRover global context = {}".format(self.__dict__))

    @property
    def user_id(self):
        return os.getenv(UserEnv.USER_ID, "")

    def pre_check_enabled(self):
        return (
            self.pre_check_operators is not None
            and len(self.pre_check_operators) > 0
        )

    def get_pre_check_operators(self):
        result_ops = []
        for op_desc in self.pre_check_operators:
            try:
                module_name = op_desc[0]
                class_name = op_desc[1]
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    result_ops.append(cls())
            except RuntimeError:
                logger.warning(
                    "Invalid pre-check "
                    f"operators: {self.pre_check_operators}"
                )
        return result_ops
