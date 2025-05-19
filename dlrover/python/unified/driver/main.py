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

import time

import ray

from dlrover.python.unified.master.mpmd.master import MPMDMaster
from dlrover.python.unified.master.spmd.master import SPMDMaster

try:
    from ray.exceptions import ActorDiedError as ade
except ImportError:
    from builtins import RuntimeError as ade

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.dl_context import DLContext, RLContext
from dlrover.python.unified.common.enums import DLType, JobStage
from dlrover.python.unified.common.exception import InvalidDLConfiguration

MASTER_CONNECT_INTERVAL = 30
MASTER_CONNECT_TIMEOUT = 3 * MASTER_CONNECT_INTERVAL


def gen_dl_context(args) -> DLContext:
    train_type = args.dl_type
    if train_type == DLType.RL:
        return RLContext.build_from_args(args)
    else:
        return DLContext.build_from_args(args)


def get_master_cls(args):
    train_type = args.dl_type
    if train_type in [DLType.RL, DLType.MULTIMODAL]:
        return MPMDMaster
    elif train_type in [DLType.SFT, DLType.PRE]:
        return SPMDMaster
    else:
        raise InvalidDLConfiguration()


def submit(args=None, blocking=True):
    # parse input arguments
    parsed_args = parse_job_args(args)

    # build job config from arguments
    job_config = JobConfig.build_from_args(parsed_args)
    logger.info(f"Job config: {job_config}")

    # build rl context from arguments
    dl_context = gen_dl_context(parsed_args)
    if not dl_context.validate():
        logger.error("DL Context is not valid.")
        raise InvalidDLConfiguration()
    logger.info(f"DL context: {dl_context}.")

    # create master actor
    name = "DLMaster-" + parsed_args.job_name
    logger.info(f"Create DLMaster for job executing: {parsed_args.job_name}.")

    runtime_env = {"env_vars": {}}
    runtime_env["env_vars"].update(dl_context.env)

    master_actor = (
        get_master_cls(parsed_args)
        .options(
            name=name,
            lifetime="detached",
            num_cpus=parsed_args.master_cpu,
            memory=parsed_args.master_mem,
            runtime_env=runtime_env,
            max_restarts=-1,
        )
        .remote(job_config.serialize(), dl_context.serialize())
    )

    ray.get(master_actor.ping.remote())
    logger.info("DLMaster created.")

    ray.get(master_actor.run.remote())
    logger.info("DLMaster is running...")

    if blocking:
        master_exit_start = 0
        while True:
            if (
                master_exit_start != 0
                and time.time() - master_exit_start > MASTER_CONNECT_TIMEOUT
            ):
                logger.warning("DLMaster might dead, exit now.")
                break

            try:
                result = ray.get(master_actor.get_job_status.remote())
                # if result in ["FINISHED", "ERROR"]:
                if JobStage.is_ending_stage(result):
                    logger.info(f"DLMaster exited with: {result}")
                    break
                master_exit_start = 0
            except ade:
                if master_exit_start == 0:
                    master_exit_start = time.time()

            logger.debug("DLMaster is running...")
            time.sleep(MASTER_CONNECT_INTERVAL)
    else:
        logger.info("Driver exit now for none blocking mode.")


def main(args=None, blocking=True):
    return submit(args=args, blocking=blocking)


if __name__ == "__main__":
    main()
