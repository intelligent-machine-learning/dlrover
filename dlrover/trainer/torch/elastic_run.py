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

import os
import sys
import telnetlib
import tempfile
import time
from typing import Callable, Union

from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing.api import SubprocessHandler
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig
from torch.distributed.run import config_from_args, get_args_parser

from dlrover.python.common.grpc import find_free_port
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import (
    GlobalMasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.training import launch_agent


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
        config: LaunchConfig,
        entrypoint: Union[Callable, str, None],
    ):
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args):
        return launch_agent(self._config, self._entrypoint, list(args))


def launch_dlrover_local_master():
    """Launch a subprocess to run the DLrover master."""
    cmd = os.getenv("PYTHON_EXEC", sys.executable)
    host = "127.0.0.1"
    port = find_free_port()
    log_dir = tempfile.mkdtemp(prefix="dlrover_master_")
    job_name = log_dir.split("_")[-1]
    stdout = os.path.join(log_dir, "stdout.log")
    stderr = os.path.join(log_dir, "stderror.log")
    logger.info(f"The master log file:\n stdout: {stdout} \n stderr: {stderr}")
    args = (
        "-u",
        "-m",
        "dlrover.python.master.main",
        "--port",
        f"{port}",
        "--job_name",
        f"standalone-{job_name}",
        "--platform",
        "local",
    )
    handler = SubprocessHandler(cmd, args, {}, stdout, stderr)
    dlrover_master_addr = f"{host}:{port}"
    while True:
        try:
            telnetlib.Telnet(host=host, port=port, timeout=3)
            logger.info("DLRover local job master starts!")
            break
        except ConnectionRefusedError:
            time.sleep(3)
    GlobalMasterClient.MASTER_CLIENT = build_master_client(dlrover_master_addr)
    return handler


def run(args):
    dlrover_master_handler = None
    if args.standalone:
        dlrover_master_handler = launch_dlrover_local_master()

    config, cmd, cmd_args = config_from_args(args)
    setattr(config, "network_check", False)
    setattr(config, "node_unit", 1)
    if hasattr(args, "network_check"):
        config.network_check = args.network_check
    if hasattr(args, "node_unit"):
        config.rdzv_configs["node_unit"] = args.node_unit
    elastic_launch(config=config, entrypoint=cmd)(*cmd_args)

    if dlrover_master_handler:
        dlrover_master_handler.close()


@record
def main(args=None):
    args = parse_args(args)
    run(args)


if __name__ == "__main__":
    main()
