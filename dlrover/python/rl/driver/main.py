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
import os

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.args import parse_job_args
from dlrover.python.rl.common.context import JobContext, RLContext
from dlrover.python.rl.common.exception import InvalidRLConfiguration
from dlrover.python.rl.master.main import DLRoverRLMaster


def submit(args):
    # parse input arguments
    parsed_args = parse_job_args(args)

    # build job config from arguments
    job_config = JobContext.build_from_args(parsed_args)
    logger.info(f"Job config: {job_config}")

    # build rl context from arguments
    rl_context = RLContext.build_from_args(parsed_args)
    if not rl_context.validate():
        logger.error("RL Context is not valid.")
        raise InvalidRLConfiguration()
    logger.info(f"RL context: {rl_context}.")

    # create master actor
    logger.info(f"Create RLMaster for job executing: {parsed_args.job_name}.")
    (
        DLRoverRLMaster.options(
            name="DLRoverRLMaster-" + parsed_args.job_name,
            lifetime="detached",
            num_cpus=parsed_args.master_cpu,
            memory=parsed_args.master_mem,
        ).remote(job_config.serialize(), rl_context.serialize())
    )


def main():
    return submit(parse_job_args())


if __name__ == "__main__":
    os._exit(main())
