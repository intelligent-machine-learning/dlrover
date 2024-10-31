# Copyright 2024 The DLRover Authors. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.launcher.api import LaunchConfig

from dlrover.python.common.constants import (
    Accelerators,
    AscendConstants,
    NodeEnv,
)
from dlrover.python.common.log import default_logger as logger


@dataclass
class ElasticLaunchConfig(LaunchConfig):
    """
    Creates a rendezvous config of elastic training.

    Args:
        network_check: whether to check the network available before training.
        comm_perf_test: whether to test the communication performance.
        node_unit: the number of unit of nodes. The number of nodes must be
            a multiple of node_unit.
        auto_config: indicate if automatically configure the nnodes and
            nproc_per_node.
        auto_tunning: whether to auto-tune the parallelism configuration.
        exclude_straggler: The node will exit if it is a straggler in network
            check and exclude_straggler is True.
        save_at_breakpoint: indicate if save the checkpoint from the shared
            memory into the disk after a failure occurs.
        accelerator: the type of accelerator processor like nvidia.com/gpu,
            ascend-npu.
        training_log_file: the training log file of this training job
        failure_node_errors: the error information that indicate the node
            is a failure node
    """

    network_check: bool = False
    comm_perf_test: bool = False
    node_unit: int = 1
    training_port: int = AscendConstants.HCCL_PORT_START_DEFAULT
    auto_config: bool = False
    auto_tunning: bool = False
    exclude_straggler: bool = False
    save_at_breakpoint: bool = False
    accelerator: str = ""
    log_dir: Optional[str] = None  # Keep Compatibility with PyTorch>=2.3.0
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    training_log_file: str = ""
    failure_node_errors: str = ""

    def set_node_unit(self, node_unit):
        """Set the number unit of nodes."""
        self.node_unit = node_unit
        self.rdzv_configs["node_unit"] = node_unit

    def auto_configure_params(self):
        self.training_log_file = os.getenv(NodeEnv.TRAINING_LOG_FILE, "")
        self.failure_node_errors = os.getenv(NodeEnv.FAILURE_NODE_ERRORS, "")
        if len(self.failure_node_errors) > 0:
            errors = self.failure_node_errors.strip()
            if errors[0] != "#" or errors[-1] != "#":
                logger.warning("invalid failure node errors: %s", errors)
                self.failure_node_errors = ""

        device = ""
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name()
        if "Ascend" in device:
            self.accelerator = Accelerators.ASCEND_NPU
        if not self.auto_config:
            return

        if NodeEnv.NODE_NUM in os.environ:
            self.min_nodes = int(os.environ[NodeEnv.NODE_NUM])
            self.max_nodes = int(os.environ[NodeEnv.NODE_NUM])
        if torch.cuda.is_available():
            self.nproc_per_node = torch.cuda.device_count()
        if self.min_nodes >= 4:
            self.network_check = True
