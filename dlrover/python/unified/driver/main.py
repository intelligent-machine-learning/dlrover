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
import os

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.config import DLConfig, JobConfig
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.controller.master import PrimeMaster


def submit(config: JobConfig, blocking=True):
    """Submit a job to DLRover."""
    # do ray init
    if ray.is_initialized():
        logger.info("Ray already initialized.")
    else:
        working_dir_env = config.dl_config.global_envs.get(
            DLWorkloadEnv.WORKING_DIR, os.getcwd()
        )
        logger.info(
            f"Using specified working dir: {working_dir_env} "
            f"instead of current working dir: {os.getcwd()}."
        )
        ray.init(runtime_env={"working_dir": working_dir_env})
        logger.info("Ray initialized.")

    master = PrimeMaster.create(config)
    logger.info("DLMaster created.")

    master.start()
    logger.info("DLMaster started.")

    if blocking:
        master.wait()
        result = master.get_status()
        logger.info(
            f"DLMaster exited: {result.stage}, exit code: {result.exit_code}"
        )
        return result.exit_code
    else:
        logger.info("Driver exit now for none blocking mode.")
        return 0


def main(args=None, blocking=True):
    """Main function to start the DLRover driver from CLI."""

    import argparse

    parser = argparse.ArgumentParser(
        description="DLRover Driver CLI to start the DLRover master."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Name of the job to be submitted.",
    )
    parser.add_argument(
        "--master_cpu",
        type=int,
        default=1,
        help="Number of CPUs for the master actor.",
    )
    parser.add_argument(
        "--master_memory",
        type=int,
        default=100,
        help="Memory (in MB) for the master actor.",
    )
    parser.add_argument(
        "--dl_config",
        type=str,
        required=True,
        help="Json for DLRover job configuration.",
    )
    args = parser.parse_args(args)
    args.dl_config = DLConfig.model_validate_json(args.dl_config)
    config = JobConfig.model_validate(args.__dict__)
    return submit(config=config, blocking=blocking)


if __name__ == "__main__":
    os._exit(main())
