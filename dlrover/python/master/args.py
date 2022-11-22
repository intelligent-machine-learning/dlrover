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
from itertools import chain

from dlrover.python.common.constants import DistributionStrategy
from dlrover.python.common.log_utils import default_logger as logger


def add_params(parser):
    add_bool_param(
        parser=parser,
        name="--use_async",
        default=False,
        help="True for asynchronous SGD, False for synchronous SGD",
    )
    add_bool_param(
        parser=parser,
        name="--need_task_manager",
        default=True,
        help="If true, master creates a task manager for dynamic sharding. "
        "Otherwise, no task manager is created",
    )
    add_bool_param(
        parser=parser,
        name="--need_node_manager",
        default=True,
        help="If true, master creates a pod manager to maintain the "
        "cluster for the job. Otherwise, no pod manager is created",
    )
    add_bool_param(
        parser=parser,
        name="--enabled_auto_ps",
        default=False,
        help="If true, the master will auto-configure the resources "
        "of PS nodes and adjust the resources at runtime",
    )
    add_bool_param(
        parser=parser,
        name="--enabled_auto_worker",
        default=False,
        help="If true, the master will auto-configure the resources "
        "of worker nodes and adjust the resources at runtime",
    )
    add_bool_param(
        parser=parser,
        name="--task_fault_tolerance",
        default=True,
        help="If true, task manager supports fault tolerance, otherwise "
        "no fault tolerance.",
    )
    add_bool_param(
        parser=parser,
        name="--relaunch_timeout_worker",
        default=True,
        help="If true, the master will detect the time of worker to "
        "execute a task and relaunch the worker if timeout",
    )
    add_bool_param(
        parser=parser,
        name="--use_ddp",
        default=False,
        help="If true, the master calls DDPRendezvousServer,"
        "or the master calls HorovodRendezvousServer",
    )
    parser.add_argument(
        "--custom_scaling_strategy",
        type=str,
        default="off",
        help="Set low priority gpu workers scaling out strategies when using "
        "gpu elastic training. If 'off', low priority gpu workers can scale "
        "out at any time as long as resources are available. If "
        "'scaling_by_time', scale out at default period of time. If "
        "'scaling_by_time:starttime-endtime',  scale out during starttime to"
        " endtime. The format of 'starttime' or 'endtime' is"
        " 'hour:minute:second' of 24-hour system. Currently, only support "
        "`scaling_by_time` strategy.",
    )
    add_bool_param(
        parser=parser,
        name="--need_tf_config",
        default=False,
        help="If true, needs to set TF_CONFIG env for ps/worker. Also "
        "need to use fixed service name for workers",
    )
    parser.add_argument(
        "--relaunch_on_worker_failure",
        type=int,
        help="The number of relaunch tries for a worker failure for "
        "PS Strategy training",
        default=3,
    )
    add_bool_param(
        parser=parser,
        name="--ps_is_critical",
        default=True,
        help="If true, ps pods are critical, and ps pod failure "
        "results in job failure.",
    )
    parser.add_argument(
        "--critical_worker_index",
        default="default",
        help="If 'default', worker0 is critical for PS strategy custom "
        "training, none for others; "
        "If 'none', all workers are non-critical; "
        "Otherwise, a list of critical worker indices such as '1:0,3:1' "
        "In each pair, the first value is the pod index and the second value "
        "is the number of allowed relaunches before becoming critical",
    )
    parser.add_argument(
        "--ps_relaunch_max_num",
        type=int,
        help="The max number of ps relaunches",
        default=1,
    )
    parser.add_argument(
        "--launch_worker_after_ps_running",
        default="default",
        help="This argument indicates if launch worker "
        "pods (execpt worker0) after all ps pods are running. "
        "If 'on', launch worker "
        "pods (execpt worker0) after all ps pods are running. "
        "If 'off', launch worker pods regardless of ps pod status "
        "If 'default', when ps.core >= 16 with PS strategy, similar "
        "to 'on', otherwise, similar to 'off'. ",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers", default=0
    )
    parser.add_argument(
        "--worker_resource_request",
        default="",
        type=str,
        help="The minimal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--worker_resource_limit",
        type=str,
        default="",
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to worker_resource_request",
    )
    parser.add_argument(
        "--num_tf_master",
        type=int,
        help="Number of TensorFlow estimator master",
        default=0,
    )
    parser.add_argument(
        "--tf_master_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by TensorFlow estimator, "
        " master e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--tf_master_resource_limit",
        type=str,
        default="",
        help="The maximal resource required by TensorFlow estimator, "
        "master e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to tf_master_resource_request",
    )
    parser.add_argument(
        "--master_pod_priority",
        default="",
        help="The requested priority of master pod",
    )
    parser.add_argument(
        "--tf_master_pod_priority",
        default="",
        help="The requested priority of tensorflow estimator master",
    )
    parser.add_argument(
        "--worker_pod_priority",
        default="",
        help="The requested priority of worker pod, we support following"
        "configs: high/low/0.5. The 0.5 means that half"
        "worker pods have high priority, and half worker pods have"
        "low priority. The default value is low",
    )
    parser.add_argument(
        "--num_ps_pods", type=int, help="Number of PS pods", default=0
    )
    parser.add_argument(
        "--ps_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--ps_resource_limit",
        default="",
        type=str,
        help="The maximal resource required by worker, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to worker_resource_request",
    )
    parser.add_argument(
        "--ps_pod_priority",
        default="",
        help="The requested priority of PS pod",
    )
    parser.add_argument(
        "--evaluator_resource_request",
        default="cpu=1,memory=4096Mi",
        type=str,
        help="The minimal resource required by evaluator, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1",
    )
    parser.add_argument(
        "--evaluator_resource_limit",
        default="",
        type=str,
        help="The maximal resource required by evaluator, "
        "e.g. cpu=1,memory=1024Mi,disk=1024Mi,gpu=1,"
        "default to evaluator_resource_request",
    )
    parser.add_argument(
        "--evaluator_pod_priority",
        default="",
        help="The requested priority of PS pod",
    )
    parser.add_argument(
        "--num_evaluators",
        type=int,
        default=0,
        help="The number of evaluator pods",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        type=str,
        help="The name of the Kubernetes namespace where ElasticDL "
        "pods will be created",
    )
    add_bool_param(
        parser=parser,
        name="--force_use_kube_config_file",
        default=False,
        help="If true, force to load the cluster config from ~/.kube/config "
        "while submitting the ElasticDL job. Otherwise, if the client is in a "
        "K8S environment, load the incluster config, if not, load the kube "
        "config file.",
    )

    parser.add_argument(
        "--distribution_strategy",
        type=str,
        choices=[
            "",
            DistributionStrategy.LOCAL,
            DistributionStrategy.PARAMETER_SERVER,
            DistributionStrategy.ALLREDUCE,
            DistributionStrategy.CUSTOM,
        ],
        default=DistributionStrategy.PARAMETER_SERVER,
        help="Master will use a distribution policy on a list of devices "
        "according to the distributed strategy, "
        "e.g. 'ParameterServerStrategy', 'AllreduceStrategy', "
        "'CustomStrategy' or 'Local'",
    )


def add_bool_param(parser, name, default, help):
    parser.add_argument(
        name,  # should be in "--foo" format
        nargs="?",
        const=not default,
        default=default,
        type=lambda x: x.lower() in ["true", "yes", "t", "y"],
        help=help,
    )


def build_arguments_from_parsed_result(args, filter_args=None):
    """Reconstruct arguments from parsed result
    Args:
        args: result from `parser.parse_args()`
    Returns:
        list of string: ready for parser to parse,
        such as ["--foo", "3", "--bar", False]
    """
    items = vars(args).items()
    if filter_args:
        items = filter(lambda item: item[0] not in filter_args, items)

    def _str_ignore_none(s):
        if s is None:
            return s
        return str(s)

    arguments = map(_str_ignore_none, chain(*items))
    arguments = [
        "--" + k if i % 2 == 0 else k for i, k in enumerate(arguments)
    ]
    return arguments


def wrap_python_args_with_string(args):
    """Wrap argument values with string
    Args:
        args: list like ["--foo", "3", "--bar", False]

    Returns:
        list of string: like ["--foo", "'3'", "--bar", "'False'"]
    """
    result = []
    for value in args:
        if not value.startswith("--"):
            result.append("'{}'".format(value))
        else:
            result.append(value)
    return result


def pos_int(arg):
    res = int(arg)
    if res <= 0:
        raise ValueError("Positive integer argument required. Got %s" % res)
    return res


def non_neg_int(arg):
    res = int(arg)
    if res < 0:
        raise ValueError(
            "Non-negative integer argument required. Get %s" % res
        )
    return res


def print_args(args, exclude_args=[], groups=None):
    """
    Args:
        args: parsing results returned from `parser.parse_args`
        exclude_args: the arguments which won't be printed.
        groups: It is a list of a list. It controls which options should be
        printed together. For example, we expect all model specifications such
        as `optimizer`, `loss` are better printed together.
        groups = [["optimizer", "loss"]]
    """

    def _get_attr(instance, attribute):
        try:
            return getattr(instance, attribute)
        except AttributeError:
            return None

    dedup = set()
    if groups:
        for group in groups:
            for element in group:
                dedup.add(element)
                logger.info("%s = %s", element, _get_attr(args, element))
    other_options = [
        (key, value)
        for (key, value) in args.__dict__.items()
        if key not in dedup and key not in exclude_args
    ]
    for key, value in other_options:
        logger.info("%s = %s", key, value)


def _build_master_args_parser():
    parser = argparse.ArgumentParser(description="ElasticDL Master")
    parser.add_argument(
        "--port",
        default=50001,
        type=pos_int,
        help="The listening port of master",
    )
    add_params(parser)
    return parser


def parse_master_args(master_args=None):
    parser = _build_master_args_parser()

    args, unknown_args = parser.parse_known_args(args=master_args)
    print_args(args)
    if unknown_args:
        logger.warning("Unknown arguments: %s", unknown_args)

    return args
