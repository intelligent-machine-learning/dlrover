# Copyright 2026 The DLRover Authors. All rights reserved.
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


"""
``dlrover-run`` provides a superset of the functionality as ``torchrun``
with the following additional functionalities:

1. Check the network of node to detect the fault node or straggler.

2. `rdzv-endpoint`, `rdzv-backend` and `rdzv-id` are not required for
multi-node multi-worker.

Usage
--------

Run in the worker Pod with GPU of ElasticJob.
++++++++++++++++++++++++++++++

::

    dlrover-run
        --auto-config
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

auto-config will set the nnodes as the number of nodes in a job,
nproc_per_node as the number of available GPUs. If the number of
nodes >= 4, it will set the network-check as True. If network-check is True,
dlrover-run will launch simple tasks on each node to check whether
the node is slow or fault.

Single-node multi-worker
++++++++++++++++++++++++++++++

::

    dlrover-run
        --standalone
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

multi-node multi-worker
+++++++++++++++++++++++++++++++++++

::

    dlrover-run
        --nnodes=$NUM_NODES
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

Elastic (``min=1``, ``max=4``, tolerates up to 3 membership
changes or failures)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    dlrover-run
        --nnodes=1:4
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

Note on rendezvous backend
------------------------------

For multi-node training you need to specify:

1. ``--network-check``: Bool, whether to check the node network to find the
    fault node or straggler.
2. ``--rdzv-conf``: We can set timeout into rdzv_conf like
    ```--rdzv-conf join_timeout=600,lastcall_timeout=60,pend_timeout=3600`.

For auto-tuning parallelism configuration, you need to specify:

1. ``--auto-tunning``: Whether to auto tune the batch size and learning rate.
"""

import importlib
import os
import re
import socket
import sys
import telnetlib
import time
import uuid
from datetime import datetime
from typing import Callable, List, Tuple, Union

from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing.api import SubprocessHandler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import launch_agent as torch_launch_agent
from torch.distributed.run import config_from_args, get_args_parser

import dlrover.python.util.common_util as cu
from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    Accelerators,
    JobConstant,
    NodeEnv,
    NodeEventType,
    PreCheckStatus,
)
from dlrover.python.common.log import (
    default_logger as logger,
    get_agent_log_dir,
)
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.elastic_agent.torch.dynamic_failover import (
    DynamicAgentFailoverExtension,
)
from dlrover.python.elastic_agent.torch.training import (
    ElasticLaunchConfig,
    launch_agent,
)
from dlrover.python.training_event import DLRoverAgentEvent
from dlrover.trainer.torch.utils import version_less_than_230


def parse_args(args):
    parser = get_args_parser()
    parser.allow_abbrev = False

    parser.add_argument(
        "--precheck",
        type=int,
        action=env,
        default=0,
        choices=[0, 1, 2],
        help="The level to check the node before starting the training task."
        "Default 0 dose not run check task; the value 1 splits nodes into "
        "groups to runs a matmul and allgather task and each group has 2 "
        "nodes; the value 2 will run an allgather task with all nodes to "
        "test the performance.",
    )
    parser.add_argument(
        "--node_unit",
        "--node-unit",
        type=int,
        action=env,
        default=1,
        help="The number unit of nodes to schedule. The scheduled number of "
        "nodes should be a multiple of node_unit.",
    )
    parser.add_argument(
        "--auto_config",
        "--auto-config",
        action=check_env,
        help="Whether to automatically configure the nnodes and nproc_per_nodes.",
    )
    parser.add_argument(
        "--auto_tunning",
        "--auto-tunning",
        action=check_env,
        help="Whether to auto-tune the parallel configuration.",
    )
    parser.add_argument(
        "--exclude-straggler",
        "--exclude_straggler",
        action=check_env,
        help="Bool, The node will exit if the node is straggler and "
        "the argument is True. The argument only works when network-check "
        "is True.",
    )
    parser.add_argument(
        "--save_at_breakpoint",
        "--save-at-breakpoint",
        action=check_env,
        help="Bool. If True, the agent in the main process will save the "
        "checkpoint in the memory to the storage if the training "
        "process fails.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        action=env,
        default=Accelerators.NVIDIA_GPU,
        choices=[
            Accelerators.NVIDIA_GPU,
            Accelerators.ASCEND_NPU,
            Accelerators.MTHREADS_GPU,
        ],
        help="The type of accelerator chip of the machine.",
    )
    parser.add_argument(
        "--training_port",
        "--training-port",
        type=int,
        action=env,
        default=60000,
        help="The start of training port.",
    )
    parser.add_argument(
        "--numa-affinity",
        "--numa_affinity",
        action=check_env,
        help="bool, set workers processes cpu numa affinity or not",
    )

    parser.add_argument(
        "--membind-policy",
        "--membind_policy",
        type=str,
        action=env,
        default="preferred",
        help="The memory bind policy, bind or preferred",
    )

    # deprecated arguments
    parser.add_argument(
        "--network-check",
        "--network_check",
        action=check_env,
        help="Whether to check network before starting training process.",
    )
    parser.add_argument(
        "--comm-perf-test",
        "--comm_perf_test",
        action=check_env,
        help="Whether to test the communication performance.",
    )

    parser.add_argument(
        "--ucp_device_type",
        "--ucp_device_type",
        action=env,
        default="cpu",
        help="The device where universal checkpoint take place.",
    )
    return parser.parse_args(args)


class ElasticLaunch:
    """
    Launches an torchelastic agent on the container
    that invoked the entrypoint.

        1. Pass the ``entrypoint`` arguments as non ``kwargs``
            (e.g. no named parameters)/
           ``entrypoint`` can be a function or a command.
        2. The return value is a map of each worker's output mapped
           by their respective global rank.

    Usage

    ::

    def worker_fn(foo):
        # ...

    def main():
        # entrypoint is a function.
        outputs = ElasticLaunch(LaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = ElasticLaunch(LaunchConfig, "script.py")(args)
        outputs = ElasticLaunch(LaunchConfig, "python")("script.py")
    """

    def __init__(
        self,
        config: ElasticLaunchConfig,
        entrypoint: Union[Callable, str, None],
        use_dlrover_launch: bool,
    ):
        self._config = config
        self._entrypoint = entrypoint
        self._use_dlrover_launch = use_dlrover_launch

    def __call__(self, *args):
        if self._use_dlrover_launch:
            wait_pre_check(self._config)
            return launch_agent(self._config, self._entrypoint, list(args))
        else:
            return torch_launch_agent(
                self._config, self._entrypoint, list(args)
            )


def wait_pre_check(config: ElasticLaunchConfig):
    """Wait master's pre-check result."""
    client = MasterClient.singleton_instance()
    if not client:
        raise RuntimeError("MasterClient is not available.")

    wait_secs = JobConstant.PRE_CHECK_WAIT_SECS

    # call master once for connection pre-check
    client.report_pre_check_status(
        NodeEventType.WAIT_PRE_CHECK, config.to_json()
    )

    while True:
        status = client.get_pre_check_result()
        if status == PreCheckStatus.PASS:
            logger.info("Pre check passed.")
            break
        elif status == PreCheckStatus.FAIL:
            logger.info("Pre check failed, training will abort...")
        elif status == PreCheckStatus.DISABLED:
            logger.info("Pre check disabled.")
            break
        else:
            logger.info(
                f"Pre check not passed yet, status: {status}, "
                f"wait for another {wait_secs}s..."
            )
        time.sleep(wait_secs)


def _launch_dlrover_local_master(master_addr, job_name):
    """Launch a subprocess to run the DLRover master."""
    logger.info(f"Start dlrover master with addr {master_addr}")
    if not master_addr:
        host = "127.0.0.1"
        port = cu.find_free_port()
    else:
        host = master_addr.split(":")[0]
        port = int(master_addr.split(":")[1])
    cmd = os.getenv("PYTHON_EXEC", sys.executable)
    args = (
        "-u",
        "-m",
        "dlrover.python.master.main",
        "--port",
        f"{port}",
        "--node_num",
        "1",
        "--job_name",
        job_name,
        "--platform",
        "local",
    )
    if version_less_than_230():
        handler = SubprocessHandler(cmd, args, {}, "", "")
    else:
        handler = SubprocessHandler(cmd, args, {}, "", "", 0)

    dlrover_master_addr = f"{host}:{port}"
    return handler, dlrover_master_addr


def _check_dlrover_master_available(addr, timeout=120):
    """Verify that the master servicer is available except ray mode."""
    if env_utils.is_ray_mode():
        logger.info("Skip dlrover master check for ray mode.")
        return True

    if not addr:
        return False

    try:
        host = addr.split(":")[0]
        port = int(addr.split(":")[1])
    except Exception:
        logger.error(f"Invalid master addr: {addr}")
        return False

    start_time = time.time()
    while True:
        try:
            telnetlib.Telnet(host=host, port=port, timeout=3)
            logger.info("DLRover master has already started.")
            return True
        except (socket.timeout, ConnectionRefusedError):
            logger.warning(
                "Got connection timeout when checking dlrover master."
            )
            time.sleep(1)
        except socket.gaierror:
            logger.warning("Got gaierror when checking dlrover master.")
            time.sleep(3)

        if time.time() - start_time > timeout:
            return False


def _elastic_config_from_args(
    args,
) -> Tuple[ElasticLaunchConfig, Union[Callable, str], List[str]]:
    config, cmd, cmd_args = config_from_args(args)

    logger.info(f"Setup ElasticLaunchConfig with: {config.__dict__}")
    elastic_config = ElasticLaunchConfig(**config.__dict__)

    elastic_config.setup_log(
        getattr(args, "log_dir", None) or get_agent_log_dir(),
        getattr(args, "redirects", None),
        getattr(args, "tee", None),
    )
    elastic_config.precheck = getattr(args, "precheck", False)
    elastic_config.network_check = getattr(args, "network_check", False)
    elastic_config.comm_perf_test = getattr(args, "comm_perf_test", False)
    elastic_config.numa_affinity = getattr(args, "numa_affinity", False)
    elastic_config.membind_policy = getattr(args, "membind_policy", "none")
    elastic_config.auto_tunning = getattr(args, "auto_tunning", False)
    elastic_config.auto_config = getattr(args, "auto_config", False)
    elastic_config.accelerator = getattr(
        args, "accelerator", Accelerators.NVIDIA_GPU
    )

    elastic_config.exclude_straggler = getattr(
        args, "exclude_straggler", False
    )
    elastic_config.set_node_unit(getattr(args, "node_unit", 1))
    elastic_config.training_port = getattr(args, "training_port", 60000)
    elastic_config.save_at_breakpoint = getattr(
        args, "save_at_breakpoint", False
    )

    _merge_elastic_config_from_master(elastic_config)

    elastic_config.auto_configure_params()
    elastic_config.update_precheck_args()
    elastic_config.rdzv_backend = "dlrover-master"
    elastic_config.rdzv_endpoint = ""
    join_timeout = elastic_config.rdzv_configs.get("join_timeout", 600)
    elastic_config.rdzv_configs["timeout"] = join_timeout

    return elastic_config, cmd, cmd_args


def _merge_elastic_config_from_master(config: ElasticLaunchConfig):
    _client = MasterClient.singleton_instance()
    try:
        logger.info("try to get elastic run config from master")
        master_configs = _client.get_elastic_run_config()
    except Exception as e:
        logger.error(f"fail to get elastic config from master: {e}")
        master_configs = {}

    # if "precheck" in master_configs:
    # logger.info("Enable precheck by master")
    # config.precheck = True

    if "network_check" in master_configs:
        logger.info("Enable network checking by master")
        config.network_check = True

    if "comm_perf_test" in master_configs:
        logger.info("Enable comm_perf_test by master")
        config.comm_perf_test = True

    if "numa_affinity" in master_configs:
        logger.info("Enable numa affinity by master")
        config.numa_affinity = True

    if "auto_tunning" in master_configs:
        logger.info("Enable auto_tunning by master")
        config.auto_tunning = True

    if "auto_config" in master_configs:
        logger.info("Enable auto_config by master")
        config.auto_config = True

    if "exclude_straggler" in master_configs:
        logger.info("Enable exclude_straggler by master")
        config.exclude_straggler = True

    if "save_at_breakpoint" in master_configs:
        logger.info("Enable save_at_breakpoint by master")
        config.save_at_breakpoint = True


def _check_to_use_dlrover_run(job_name, is_standalone=False):
    """
    Standalone mode:
        1) dlrover-run with local master
        2) torchrun without dlrover master

    Distributed mode:
        dlrover-run with distributed master

    Notice: 'torchrun' is not supported in dlrover with distributed mode.
    So user should use 'torchrun' directly(without 'dlrover-run') to run
    distributed training if no dlrover available.
    """
    master_addr = os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "")
    node_rank = env_utils.get_node_rank()

    # try dist master connection
    dist_master_available = _check_dlrover_master_available(
        master_addr, timeout=60
    )

    if not dist_master_available:
        if is_standalone:
            # for standalone mode
            if node_rank == 0:
                # create local master
                master_handler, master_addr = _launch_dlrover_local_master(
                    master_addr,
                    job_name,
                )
                logger.info(
                    f"Set the dlrover master(local) addr as {master_addr}"
                )
                os.environ[NodeEnv.DLROVER_MASTER_ADDR] = master_addr

                # try local master connection
                if not _check_dlrover_master_available(
                    master_addr, timeout=30
                ):
                    logger.warning(
                        "Downgrade to use torch-run in standalone for "
                        "local dlrover master is unavailable."
                    )
                    # torch-run(standalone)
                    return False, None
                else:
                    # dlrover-run + local-master(standalone)
                    return True, master_handler
            else:
                # raise exception directly
                raise RuntimeError(
                    "Only single node is supported in standalone mode."
                )
        else:
            # for distribution mode
            # raise exception directly
            raise RuntimeError(
                "Distributed dlrover master is unavailable for distribution."
            )
    else:
        if is_standalone:
            logger.info(
                "Use distributed mode instead of standalone mode for "
                "distributed dlrover master is available"
            )

        # dlrover-run + dist-master(distributed)
        return True, None


def _setup_dynamic_failover_extension(config: ElasticLaunchConfig):
    extension_config = env_utils.get_env(
        NodeEnv.DLROVER_EXTENSION_DYNAMIC_FAILOVER
    )
    if not extension_config:
        return

    pattern = r"^([^:]+?(?:\.[^:]+)*)::(\w+)$"
    match = re.match(pattern, extension_config)
    if not match:
        logger.warning(
            f"User's extension config for dynamic-failover is not valid: {extension_config}."
        )
        return
    module_path = match.group(1)
    class_name = match.group(2)

    # import module and class
    try:
        module = importlib.import_module(module_path)
        extension_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"Failed to import dynamic failover extension class {class_name} from {module_path}: {e}"
        )
        return

    if not issubclass(extension_class, DynamicAgentFailoverExtension):
        logger.warning(
            f"{class_name} must inherit from DynamicAgentFailoverExtension"
        )
        return
    else:
        config.dynamic_failover_extension = extension_class()
        logger.info(f"Dynamic failover extension is setup: {extension_class}")


def run(args):
    # export event for dlrover agent
    agent = DLRoverAgentEvent.singleton_instance()
    agent.start(pid=vars(args))

    logger.info(f"DLRover agent started with: {cu.get_dlrover_version()}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_name = os.getenv(NodeEnv.JOB_NAME, f"standalone_{timestamp}")
    os.environ[NodeEnv.TORCHELASTIC_RUN_ID] = job_name

    is_standalone = args.standalone
    logger.info(f"Standalone mode: {is_standalone}")
    use_dlrover_launch, master_handler = _check_to_use_dlrover_run(
        job_name, is_standalone
    )

    # for torchrun standalone mode
    if is_standalone and not use_dlrover_launch:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:29400"
        args.rdzv_id = str(uuid.uuid4())
        logger.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv-backend={args.rdzv_backend} "
            f"--rdzv-endpoint={args.rdzv_endpoint} "
            f"--rdzv-id={args.rdzv_id}\n"
            f"**************************************\n"
        )
    config, cmd, cmd_args = _elastic_config_from_args(args)
    config.run_id = job_name
    config.role = "dlrover-trainer"

    # load custom extension:
    # dynamic failover extension
    _setup_dynamic_failover_extension(config)

    try:
        ElasticLaunch(
            config=config,
            entrypoint=cmd,
            use_dlrover_launch=use_dlrover_launch,
        )(*cmd_args)
    finally:
        if master_handler:
            master_handler.close()


@record
def main(args=None):
    args = parse_args(args)
    run(args)


if __name__ == "__main__":
    main()
