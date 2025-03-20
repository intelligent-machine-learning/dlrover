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
from util.common_util import print_args

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    TrainerArcType,
    TrainerType,
)


def _parse_trainer_type(value: str):
    try:
        return TrainerType(value.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid trainer type: {value}. "
            f"Expected one of {[a.value for a in TrainerType]}"
        )


def _parse_algorithm_type(value: str):
    try:
        return RLAlgorithmType(value.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid algorithm type: {value}. "
            f"Expected one of {[a.value for a in RLAlgorithmType]}"
        )


def _parse_trainer_arc_type(value: str):
    try:
        return TrainerArcType(value.upper())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid trainer arc type: {value}. "
            f"Expected one of {[a.value for a in TrainerArcType]}"
        )


def _parse_omega_config(value: Union[str, Dict[str, Any]]):
    try:
        return OmegaConf.create(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid trainer config: {value}. Expected JSON/DICT format."
        )


def _build_rl_args_parser():
    parser = argparse.ArgumentParser(description="RL Training")
    parser.add_argument(
        "--rl_config",
        "--rl-config",
        default={},
        type=_parse_omega_config,
        help='The full configurations for rl trainer in JSON/DICT format: {"trainer_type":"USER_DEFINED / OPENRLHF_DEEPSPEED / ...","trainer_arc_type":"MEGATRON / FSDP / ...","algorithm_type":"GRPO / PPO / ...","config":{},"workload":{"actor":{"num":"n","module":"xxx","class":"xxx"},"generator":{"num":"n","module":"xxx","class":"xxx"},"reference":{"num":"n","module":"xxx","class":"xxx"},"reward":{"num":"n","module":"xxx","class":"xxx"},"critic":{"num":"n","module":"xxx","class":"xxx"}}}',
    )
    return parser


def parse_rl_args(args=None):
    parser = _build_rl_args_parser()

    args, unknown_args = parser.parse_known_args(args=args)
    print_args(args)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    return args
