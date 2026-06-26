# Copyright 2025 The DLRover Authors. All rights reserved.
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

"""Subprocess entry point for DistributedJobMaster with simulation stubs.

This module mirrors dlrover/python/master/main.py for the K8s / distributed
platform, but replaces all Kubernetes dependencies with no-op stubs so that
the master can run on a single machine without a real cluster.

Usage (via MasterContext with master_type="dist"):

    python -m dlrover.python.testing.master.sim_master_main
        --port   <port>
        --node_num <num_workers>
        --job_name <name>
        [--namespace <ns>]
        [--max_relaunch_count <n>]

The startup sequence mirrors main.py:
    job_args.initilize()
    DistributedJobMaster(port, job_args)
    master.prepare()
    master.pre_check()
    master.run()           # blocks until agents finish or the process is killed
"""

import argparse
import os
from unittest.mock import patch

from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.args import parse_master_args
from dlrover.python.testing.master.sim_stubs import (
    SimElasticJob,
    SimJobArgs,
    SimNodeWatcher,
    SimScaler,
)

_dlrover_context = Context.singleton_instance()

# Names of the K8s factory symbols inside dist_job_manager that we must patch.
_DIST_JM = "dlrover.python.master.node.dist_job_manager"

_PATCHES = [
    # create_job_manager() calls these three factory functions.
    patch(f"{_DIST_JM}.new_elastic_job"),
    patch(f"{_DIST_JM}.new_node_watcher"),
    patch(f"{_DIST_JM}.new_job_scaler"),
    # DistributedJobManager.__init__ calls this directly (result is stored but
    # never read, so None is safe).
    patch(f"{_DIST_JM}.new_scale_plan_watcher", return_value=None),
]


def _parse_extra_args():
    """Accept --max_relaunch_count on top of the standard master args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_relaunch_count", type=int, default=3)
    extra, _ = parser.parse_known_args()
    return extra


def main():
    args = parse_master_args()
    extra = _parse_extra_args()

    _dlrover_context.config_master_port(port=args.port)
    # ALLREDUCE always relaunches; mirrors update_context() in main.py.
    _dlrover_context.relaunch_always = True
    _dlrover_context.print_config()

    job_args = SimJobArgs(
        num_workers=args.node_num,
        max_relaunch_count=extra.max_relaunch_count,
        job_name=args.job_name,
        namespace=args.namespace,
    )
    job_args.initilize()

    sim_job = SimElasticJob()
    sim_watcher = SimNodeWatcher()
    sim_scaler = SimScaler()

    # Wire stub return values into the three factory patches.
    _PATCHES[0].return_value = sim_job  # new_elastic_job  → SimElasticJob
    _PATCHES[1].return_value = sim_watcher  # new_node_watcher → SimNodeWatcher
    _PATCHES[2].return_value = sim_scaler  # new_job_scaler   → SimScaler
    # _PATCHES[3] already has return_value=None for new_scale_plan_watcher.

    logger.info(
        f"Starting DistributedJobMaster (sim) at port={args.port} "
        f"workers={args.node_num} job={args.job_name}"
    )

    # Apply all patches simultaneously so that DistributedJobMaster and
    # DistributedJobManager see the stubs when they call the factory functions.
    for p in _PATCHES:
        p.start()
    try:
        from dlrover.python.master.dist_master import DistributedJobMaster

        master = DistributedJobMaster(args.port, job_args)
        master.prepare()
        master.pre_check()
        return master.run()
    finally:
        for p in _PATCHES:
            p.stop()


if __name__ == "__main__":
    os._exit(main() or 0)
