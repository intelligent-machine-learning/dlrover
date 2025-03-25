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

import argparse
from typing import Any, Dict, Union

from omegaconf import OmegaConf

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)
from dlrover.python.util.args_util import parse_dict, pos_int
from dlrover.python.util.common_util import print_args


def _parse_master_state_backend_type(value: str):
    try:
        return MasterStateBackendType(value.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid master state backend type: {value}. "
            f"Expected one of {[a.value for a in MasterStateBackendType]}"
        )


def _parse_scheduling_strategy_type(value: str):
    try:
        return SchedulingStrategyType(value.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid scheduling strategy type: {value}. "
            f"Expected one of {[a.value for a in SchedulingStrategyType]}"
        )


def _parse_omega_config(value: Union[str, Dict[str, Any]]):
    try:
        return OmegaConf.create(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid trainer config: {value}. Expected JSON/DICT format."
        )


def _build_job_args_parser():
    parser = argparse.ArgumentParser(description="RL Training")
    parser.add_argument(
        "--job_name",
        help="ElasticJob name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--master_cpu",
        default=2,
        type=pos_int,
        help="The number of cpu for the dlrover rl master actor. "
        "Default is 2.",
    )
    parser.add_argument(
        "--master_mem",
        default=4096,
        type=pos_int,
        help="The size of memory(mb) for the dlrover rl master actor. "
        "Default is 4096.",
    )
    parser.add_argument(
        "--master_state_backend_type",
        "--state_backend_type",
        default=MasterStateBackendType.RAY_INTERNAL.value,
        type=_parse_master_state_backend_type,
        help="The state backend type for the dlrover rl master actor. "
        "Default is 'RAY_INTERNAL'.",
    )
    parser.add_argument(
        "--master_state_backend_config",
        "--state_backend_config",
        default={},
        type=parse_dict,
        help="The state backend configuration for the dlrover rl "
        "master actor. Default is {}.",
    )
    parser.add_argument(
        "--scheduling_strategy_type",
        "--scheduling_strategy_type",
        default=SchedulingStrategyType.SIMPLE.value,
        type=_parse_scheduling_strategy_type,
        help="The scheduling strategy type for the dlrover rl master to "
        "create workloads.",
    )
    parser.add_argument(
        "--rl_config",
        default={},
        type=_parse_omega_config,
        help="The full configurations for rl trainer in JSON/DICT format: "
        '{"trainer_type":"USER_DEFINED / OPENRLHF_DEEPSPEED / ...",'
        '"trainer_arc_type":"MEGATRON / FSDP / ...","algorithm_type":'
        '"GRPO / PPO / ...","config":{},"workload":{"actor":{"num":"n",'
        '"module":"xxx","class":"xxx"},"generator":{"num":"n","module":'
        '"xxx","class":"xxx"},"reference":{"num":"n","module":"xxx",'
        '"class":"xxx"},"reward":{"num":"n","module":"xxx",'
        '"class":"xxx"},"critic":{"num":"n","module":"xxx",'
        '"class":"xxx"}}}',
        required=True,
    )
    return parser


def parse_job_args(args=None):
    parser = _build_job_args_parser()

    args, unknown_args = parser.parse_known_args(args=args)
    print_args(args)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    return args
