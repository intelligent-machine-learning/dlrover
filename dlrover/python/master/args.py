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

from dlrover.python.common.global_context import DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.util.args_util import parse_tuple_list, pos_int
from dlrover.python.util.common_util import print_args


def _build_master_args_parser():
    parser = argparse.ArgumentParser(description="Training Master")
    parser.add_argument("--job_name", help="ElasticJob name", required=True)
    parser.add_argument(
        "--namespace",
        default="default",
        type=str,
        help="The name of the Kubernetes namespace where ElasticJob "
        "pods will be created",
    )
    parser.add_argument(
        "--platform",
        default="pyk8s",
        type=str,
        help="The name of platform which can be pyk8s, k8s, ray or local.",
    )
    parser.add_argument(
        "--pending_timeout",
        "--pending-timeout",
        default=DefaultValues.SEC_TO_WAIT_PENDING_POD,
        type=int,
        help="The timeout value of pending.",
    )
    parser.add_argument(
        "--pending_fail_strategy",
        "--pending-fail-strategy",
        default=DefaultValues.PENDING_FAIL_STRATEGY,
        type=int,
        help="The fail strategy for pending case. "
        "Options: -1: disabled; 0: skip; 1: necessary part; 2: all",
    )
    parser.add_argument(
        "--service_type",
        "--service-type",
        default="grpc",
        type=str,
        help="The service type of master: grpc/http.",
    )
    parser.add_argument(
        "--pre_check_ops",
        "--pre-check-ops",
        default=DefaultValues.PRE_CHECK_OPS,
        type=parse_tuple_list,
        help="The pre-check operators configuration, "
        "format: [(${module_name}, ${class_name}, ${boolean}), ...]. "
        "The boolean value represent 'bypass or not'. If set to False "
        "it indicates a bypass, otherwise it indicates normal execution.",
    )
    parser.add_argument(
        "--port",
        default=0,
        type=pos_int,
        help="The listening port of master",
    )
    parser.add_argument(
        "--node_num",
        default=1,
        type=pos_int,
        help="The number of nodes",
    )
    parser.add_argument(
        "--hang_detection",
        default=1,
        type=pos_int,
        help="The strategy of 'hang detection', "
        "0: log only; 1: notify; 2: with fault tolerance",
    )
    parser.add_argument(
        "--hang_downtime",
        default=5,
        type=pos_int,
        help="Training downtime to detect job hang, unit is minute",
    )
    parser.add_argument(
        "--xpu_type",
        default="nvidia",
        type=str,
        help="The type of XPU, should be 'nvidia' or 'ascend'",
    )
    parser.add_argument(
        "--task_process_timeout",
        default=DefaultValues.SEC_TO_TIMEOUT_TASK_PROCESS,
        type=pos_int,
        help="The timeout value of worker task process(For PS type job).",
    )
    parser.add_argument(
        "--training_elastic_mode",
        default=DefaultValues.TRAINING_ELASTIC_MODE,
        type=str,
        help="The training elastic mode: base or ucp.",
    )
    parser.add_argument(
        "--dynamic_failover_extension",
        default=None,
        type=str,
        help="Users can inject custom fault tolerance logic through this parameter. "
        "The argument format is 'module::class'. The class should implement "
        "'dlrover.python.elastic_agent.torch.dynamic_failover::DynamicAgentFailoverExtension'.",
    )
    return parser


def parse_master_args(master_args=None):
    parser = _build_master_args_parser()

    args, unknown_args = parser.parse_known_args(args=master_args)
    print_args(args)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    return args
