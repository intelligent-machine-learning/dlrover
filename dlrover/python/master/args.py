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
from dlrover.python.util.args_util import (
    parse_group_affinity,
    parse_tuple_list,
    pos_int,
)
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
        "--max_hang_downtime",
        default=DefaultValues.MAX_HANG_DOWNTIME,
        type=pos_int,
        help="The max downtime to detect job hang for non-pure-train step "
        "(e.g. train,inspect / eval) and the first step, unit is minute. "
        "Should be no less than --hang_downtime.",
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
    parser.add_argument(
        "--enable_dashboard",
        default="false",
        type=str,
        help="Enable the DLRover dashboard for job monitoring. "
        "Value should be 'true' or 'false'.",
    )
    parser.add_argument(
        "--dashboard_port",
        default=8080,
        type=pos_int,
        help="The port of the DLRover dashboard.",
    )
    parser.add_argument(
        "--group-affinity",
        "--group_affinity",
        default=None,
        type=parse_group_affinity,
        help='Node group sizes, e.g. --group-affinity="{0: 10, 1: 15}" '
        "means group 0 has 10 pods and group 1 has 15 pods. The worker "
        "replicas in the ElasticJob CRD must equal the sum of all group "
        "sizes, otherwise the master fails to start.",
    )
    parser.add_argument(
        "--node-group-strategy",
        "--node_group_strategy",
        default=None,
        choices=["ep_dp_pp", "ep_pp_dp"],
        help="Strategy used to map worker nodes (by creation order) into "
        "node groups (== physical segments). 'ep_dp_pp' keeps the existing "
        "behavior: groups occupy contiguous rank ranges. 'ep_pp_dp' packs "
        "EP-group slots of a Megatron MoE pipeline stage into ep-pp columns "
        "across groups so EP groups and PP stay intra-segment while only "
        "the dense DP collective crosses segments. Required when "
        "--group-affinity or --soft-group-affinity is configured; setting "
        "a strategy without any group affinity is rejected. The parallel "
        "sizes below are consulted only when a node-group affinity is "
        "configured.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        "--tensor_model_parallel_size",
        "--tp",
        default=1,
        type=pos_int,
        help="Tensor-parallel size. Recorded on the job args but not used "
        "by the node-group scheduling today.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        "--pipeline_model_parallel_size",
        "--pp",
        default=1,
        type=pos_int,
        help="Pipeline-parallel size. Consulted when a node-group affinity "
        "is configured.",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        "--expert_model_parallel_size",
        "--ep",
        default=1,
        type=pos_int,
        help="Expert-model-parallel size. Consulted when a node-group "
        "affinity is configured; the EP group size is --etp * --ep.",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        "--expert_tensor_parallel_size",
        "--etp",
        default=None,
        type=pos_int,
        help="Expert-tensor-parallel size. When unset it inherits "
        "--tensor-model-parallel-size. When set explicitly it must be "
        "used together with --expert-model-parallel-size; the real EP "
        "group size is expert-tensor-parallel-size * "
        "expert-model-parallel-size. Consulted when a node-group affinity "
        "is configured.",
    )
    parser.add_argument(
        "--context-parallel-size",
        "--context_parallel_size",
        "--cp",
        default=1,
        type=pos_int,
        help="Context-parallel size. Recorded on the job args but not used "
        "by the node-group scheduling today.",
    )
    parser.add_argument(
        "--soft-group-affinity",
        "--soft_group_affinity",
        default=None,
        type=parse_group_affinity,
        help="Unequal-size node group pod counts, e.g. "
        '--soft-group-affinity="{0: 30, 1: 20}" means group 0 schedules '
        "30 pods and group 1 schedules 20 pods. Unlike --group-affinity "
        "the sizes may be heterogeneous, enabling --node-group-strategy "
        "placement over uneven segments. The worker replicas must equal "
        "the sum of all group sizes. Mutually exclusive with "
        "--group-affinity.",
    )
    parser.add_argument(
        "--no-group-failover",
        "--no_group_failover",
        action="store_true",
        default=False,
        help="With --soft-group-affinity: drop the node-group labels of "
        "a relaunched (failover) worker pod so it can be scheduled onto "
        "any segment. Defaults to false, i.e. the relaunch keeps its "
        "group and follows the original segment affinity.",
    )
    parser.add_argument(
        "--enable-topology-rerank",
        "--enable_topology_rerank",
        action="store_true",
        default=False,
        help="Reorder the elastic-training rendezvous world by the psw "
        "network topology observed at rendezvous time (one node group "
        "per distinct psw from the GroupTopologyQuerier, arbitrary group "
        "sizes) so EP/PP communication groups stay intra-psw as much as "
        "possible. Requires --node-group-strategy and the parallel "
        "sizes; mutually exclusive with --group-affinity and "
        "--soft-group-affinity. Only the world ORDER changes (node_rank "
        "values stay the stable pod identity).",
    )
    return parser


def parse_master_args(master_args=None):
    parser = _build_master_args_parser()

    args, unknown_args = parser.parse_known_args(args=master_args)
    print_args(args)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    return args
