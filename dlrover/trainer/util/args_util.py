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

import argparse

from dlrover.trainer.util.log_util import default_logger as logger


def add_platform_args(parser):
    parser.add_argument(
        "--platform",
        help="execute platform",
        type=lambda x: x.upper(),
        default="KUBERNETES",
    )
    parser.add_argument(
        "--worker_action", help="worker's action", default="run"
    )

    parser.add_argument("--ps_num", help="ps number", type=int, default=1)

    parser.add_argument(
        "--worker_num", help="worker number", type=int, default=3
    )

    parser.add_argument(
        "--evaluator_num", help="evaluator number", type=int, default=1
    )

    parser.add_argument(
        "--conf", help="configuration for training", default=None
    )

    parser.add_argument("--task_id", help="worker id", type=int)

    parser.add_argument("--task_type", help="worker type", type=str)

    parser.add_argument("--mock", type=bool)

    parser.add_argument(
        "--enable_auto_scaling",
        help="configuration for elastic training",
        type=bool,
        default=False,
    )


def build_parser():
    """Build a parser for dlrover trainer"""
    parser = argparse.ArgumentParser()
    add_platform_args(parser)
    return parser


def get_parsed_args():
    """Get parsed arguments"""
    parser = build_parser()
    args, _ = parser.parse_known_args()
    if _:
        logger.info("Unknown arguments: %s", _)
    return args
