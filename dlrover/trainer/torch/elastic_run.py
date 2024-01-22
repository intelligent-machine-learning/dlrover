# Copyright 2023 The DLRover Authors. All rights reserved.
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

    torchrun
        --nnodes=$NUM_NODES
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

Elastic (``min=1``, ``max=4``, tolerates up to 3 membership
changes or failures)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
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

import os
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
from torch.distributed.run import (
    config_from_args,
    get_args_parser,
    parse_min_max_nnodes,
)

from dlrover.python.common import env_utils, grpc
from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.torch.training import (
    ElasticLaunchConfig,
    launch_agent,
)


def parse_args(args):
    parser = get_args_parser()
    parser.add_argument(
        "--network-check",
        "--network_check",
        action=check_env,
        help="Whether to check network before starting training process.",
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
        "--auto_tunning",
        "--auto-tunning",
        action=check_env,
        help="Whether to auto-tune the parallel configuraion.",
    )
    parser.add_argument(
        "--exclude-straggler",
        "--exclude_straggler",
        action=check_env,
        help="Bool, The node will exit if the node is straggler and "
        "the argument is True. The argument only works when network-check "
        "is True.",
    )
    return parser.parse_args(args)


class elastic_launch:
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
        outputs = elastic_launch(LaunchConfig, worker_fn)(foo)
        # return rank 0's output
        return outputs[0]

        # entrypoint is a command and ``script.py`` is the python module.
        outputs = elastic_launch(LaunchConfig, "script.py")(args)
        outputs = elastic_launch(LaunchConfig, "python")("script.py")
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
            return launch_agent(self._config, self._entrypoint, list(args))
        else:
            return torch_launch_agent(
                self._config, self._entrypoint, list(args)
            )


def _launch_dlrover_local_master(master_addr, job_name, node_num):
    """Launch a subprocess to run the DLrover master."""
    logger.info(f"Start dlrover master with addr {master_addr}")
    if not master_addr:
        host = "127.0.0.1"
        port = grpc.find_free_port()
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
        f"{node_num}",
        "--job_name",
        job_name,
        "--platform",
        "local",
    )
    handler = SubprocessHandler(cmd, args, {}, "", "")
    dlrover_master_addr = f"{host}:{port}"
    return handler, dlrover_master_addr


def _check_dlrover_master_available(addr, timeout=120):
    """Verify that the master grpc servicer is available."""
    host = addr.split(":")[0]
    port = int(addr.split(":")[1])
    start_time = time.time()
    while True:
        try:
            telnetlib.Telnet(host=host, port=port, timeout=3)
            logger.info("DLRover job master starts!")
            return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1)

        if time.time() - start_time > timeout:
            return False


def _elastic_config_from_args(
    args,
) -> Tuple[ElasticLaunchConfig, Union[Callable, str], List[str]]:
    config, cmd, cmd_args = config_from_args(args)
    elastic_config = ElasticLaunchConfig(**config.__dict__)
    elastic_config.network_check = getattr(args, "network_check", False)
    elastic_config.auto_tunning = getattr(args, "auto_tunning", False)
    elastic_config.exclude_straggler = getattr(
        args, "exclude_straggler", False
    )
    elastic_config.set_node_unit(getattr(args, "node_unit", 1))
    return elastic_config, cmd, cmd_args


def run(args):
    master_handler = None
    master_addr = os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "")
    use_dlrover_launch = False
    node_rank = env_utils.get_node_rank()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_name = os.getenv(NodeEnv.JOB_NAME, f"standalone_{timestamp}")
    os.environ[NodeEnv.TORCHELASTIC_RUN_ID] = job_name
    dlrover_master_ready = grpc.addr_connected(master_addr)
    if not dlrover_master_ready and node_rank == 0:
        # Only start the dlrover master on the rank-0 node.
        _, max_nodes = parse_min_max_nnodes(args.nnodes)
        master_handler, master_addr = _launch_dlrover_local_master(
            master_addr,
            job_name,
            max_nodes,
        )
        os.environ[NodeEnv.DLROVER_MASTER_ADDR] = master_addr
    if _check_dlrover_master_available(master_addr):
        use_dlrover_launch = True
    else:
        use_dlrover_launch = False

    if args.standalone and not use_dlrover_launch:
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
    try:
        elastic_launch(
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
