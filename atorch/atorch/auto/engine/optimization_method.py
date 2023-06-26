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

from atorch.common.log_utils import default_logger as logger


class OptimizationMethod(object):
    """Optimization method info
    name: str
    group: str
    supported_devices: list[str]
    min_sm_version: None or str. None for no min sm version.
        str is in "x.y" format such as amp's min sm version is "6.0"
    opt_computation/opt_memory/opt_communication: None or bool.
        True for speedup computation/descrease memory/speedup communication;
        False for negative effect; None for no effect or not determined.
    distributed_only: bool if only for distributed training
    process_mode: "ONE_PROCESS" or "ALL_PROCESS"
    is_tunable: bool
    disabled: bool for if this method is disabled for current acceleration.
    """

    def __init__(
        self,
        name,
        group,
        supported_devices=["cpu", "cuda"],
        min_sm_version=None,
        opt_computation=None,
        opt_memory=None,
        opt_communication=None,
        distributed_only=False,
        process_mode="ONE_PROCESS",
        is_tunable=True,
    ):
        self.name = name
        self.group = group
        self.supported_devices = supported_devices
        self.min_sm_version = min_sm_version
        self.opt_computation = opt_computation
        self.opt_memory = opt_memory
        self.opt_communication = opt_communication
        self.distributed_only = distributed_only
        self.process_mode = process_mode
        self.is_tunable = is_tunable
        self.disabled = False


class OptimizationMethodLibrary(object):
    """
    The supported optimization method for current acceleration engine
    Support query by method name and group.
    Strategy generation algorithms may add optimization method info here.
    """

    def __init__(self):
        self.methods = {}
        self.groups = {}
        self.invalid_combinations = []
        self.generate_group_info()
        self.add_methods()
        self.init_invalid_combinations()

    def add_methods(self):
        for name in self.groups["amp"]:
            method = OptimizationMethod(
                name=name,
                group="amp",
                supported_devices=["cuda"],
                min_sm_version="6.0",
                opt_computation=True,
                process_mode="ONE_PROCESS",
                is_tunable=False,
            )
            self.methods[method.name] = method
        for name in self.groups["zero"]:
            method = OptimizationMethod(
                name=name,
                group="zero",
                supported_devices=["cuda"] if name == "fsdp" else ["cpu", "cuda"],
                min_sm_version=None,
                opt_memory=True,
                distributed_only=True,
                process_mode="ONE_PROCESS",
                is_tunable=False,
            )
            self.methods[method.name] = method

        for name in self.groups["parallel"]:
            method = OptimizationMethod(
                name=name,
                group="parallel",
                supported_devices=["cuda"],
                min_sm_version=None,
                opt_memory=True,
                distributed_only=True,
                process_mode="ONE_PROCESS",
                is_tunable=True,
            )
            self.methods[method.name] = method

        for name in self.groups["module_replace"]:
            method = OptimizationMethod(
                name=name,
                group="module_replace",
                supported_devices=["cuda"],
                min_sm_version="6.0",
                opt_computation=True,
                opt_memory=True,
                distributed_only=False,
                process_mode="ONE_PROCESS",
                is_tunable=False,
            )
            self.methods[method.name] = method

        for name in self.groups["checkpoint"]:
            method = OptimizationMethod(
                name=name,
                group="checkpoint",
                supported_devices=["cuda"],
                min_sm_version=None,
                opt_computation=False,
                opt_memory=True,
                distributed_only=False,
                process_mode="ONE_PROCESS",
                is_tunable=False,
            )
            self.methods[method.name] = method

        # parallel_mode is a special method to support
        # 1. create process groups
        # 2. support data parallel
        method = OptimizationMethod(
            name="parallel_mode",
            group="parallel_mode",
            supported_devices=["cpu", "cuda"],
            min_sm_version=None,
            distributed_only=True,
            process_mode="ONE_PROCESS",
            is_tunable=False,
        )
        self.methods[method.name] = method

    def disable_opts(self, names):
        """
        Set `OptimizationMethod.disabled` to True according to names.
        Args:
            names(list): a list of opt method names or group names
        """
        for name in names:
            # a group name
            if name in self.groups:
                self.disable_opt_methods(self.groups[name])
            # a opt method name
            else:
                self.disable_opt_methods([name])

    def disable_opt_methods(self, names):
        """
        Disable opt method
        Args:
            names(list): opt method names
        """
        for name in names:
            self.methods[name].disabled = True

    def generate_group_info(self):
        self.groups["amp"] = ["amp_native", "amp_apex_o1", "amp_apex_o2"]
        self.groups["zero"] = ["zero1", "zero2", "fsdp"]
        self.groups["parallel_mode"] = ["parallel_mode"]
        self.groups["parallel"] = [
            "tensor_parallel",
            "pipeline_parallel",
            "mixed_parallel",
        ]
        self.groups["module_replace"] = ["module_replace"]
        self.groups["checkpoint"] = ["checkpoint"]

    def init_invalid_combinations(self):
        # TODO: add invalid combination info
        pass

    def validate_strategy(self, strategy):
        """validate strategy, and return (if_valid, process_mode)"""
        status = True
        process_mode = "ONE_PROCESS"
        for opt_method in strategy:
            if len(opt_method) != 3:
                logger.warning(
                    f"an optimization method should be a 3-element-tuple, "
                    f"but got {opt_method}. Set status to False."
                )
                status = False
            name, _, _ = opt_method
            if name not in self.methods.keys() or self.methods[name].disabled:
                status = False
                break
            if self.methods[name] == "ALL_PROCESS":
                process_mode = "ALL_PROCESS"
        # TODO: check if strategy has invalid combinations
        return status, process_mode
