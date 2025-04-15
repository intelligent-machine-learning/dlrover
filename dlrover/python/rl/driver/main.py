# Copyright 2025 The EasyDL Authors. All rights reserved.
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

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.exception import InvalidRLConfiguration
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.main import DLRoverRLMaster


def submit(args=None):
    # parse input arguments
    parsed_args = parse_job_args(args)

    # build job config from arguments
    job_config = JobConfig.build_from_args(parsed_args)
    logger.info(f"Job config: {job_config}")

    # build rl context from arguments
    rl_context = RLContext.build_from_args(parsed_args)
    if not rl_context.validate():
        logger.error("RL Context is not valid.")
        raise InvalidRLConfiguration()
    logger.info(f"RL context: {rl_context}.")

    # create master actor
    name = "DLRoverRLMaster-" + parsed_args.job_name
    logger.info(f"Create RLMaster for job executing: {parsed_args.job_name}.")

    runtime_env = {"env_vars": {}}
    runtime_env["env_vars"].update(rl_context.env)

    master_actor = DLRoverRLMaster.options(
        name=name,
        lifetime="detached",
        num_cpus=parsed_args.master_cpu,
        memory=parsed_args.master_mem,
        runtime_env=runtime_env,
    ).remote(job_config.serialize(), rl_context.serialize())

    ray.get(master_actor.ping.remote())
    logger.info("RLMaster created.")

    ray.get(master_actor.run.remote())
    logger.info("RLMaster is running.")

    logger.info("Driver exit now.")


def main(args=None):
    print(type(DLRoverRLMaster))
    return submit(args=args)


if __name__ == "__main__":
    main()
