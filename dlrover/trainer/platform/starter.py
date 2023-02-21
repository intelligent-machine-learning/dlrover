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

import os
import traceback

from dlrover import trainer
from dlrover.trainer.constants.platform_constants import PlatformConstants
from dlrover.trainer.mock.tf_process_scheduler import (
    TFProcessScheduler,
    mock_k8s_platform_subprocess,
    mock_ray_platform_subprocess,
)
from dlrover.trainer.util.args_util import get_parsed_args
from dlrover.trainer.util.log_util import default_logger as logger
from dlrover.trainer.worker.tf_kubernetes_worker import TFKubernetesWorker
from dlrover.trainer.worker.tf_ray_worker import TFRayWorker


def print_info(append_detail=False):
    """Print dlrover_trainer information"""
    dlrover_trainer = os.path.dirname(trainer.__file__)
    file_path = os.path.join(dlrover_trainer, "COMMIT_INFO")
    if not os.path.exists(file_path):
        logger.info("Whl is not built by sh build.sh, please be careful.")
        return
    with open(file_path, encoding="utf-8") as fd:
        commit_id = fd.readline().strip()
        user = fd.readline().strip()
        time = fd.readline().strip()
    logger.info(trainer.logo_string)
    logger.info("-" * 30)
    logger.info("Penrose version: %s", trainer.__version__)
    logger.info("Build by: %s", user)
    logger.info("Build time: %s", time)
    logger.info("Commit id: %s", commit_id)
    logger.info("Pid: %s", os.getpid())
    logger.info("CWD: %s", os.getcwd())
    logger.info("-" * 30)


def execute(args):
    """run routine"""
    platform = args.platform.upper()
    if platform == PlatformConstants.Kubernetes():
        worker = TFKubernetesWorker(args)
    if platform == PlatformConstants.Ray():
        worker = TFRayWorker(args)
    elif PlatformConstants.Local() in platform:
        # local mode, actually we use a scheduler
        logger.info("create ProcessScheduler with run_type = ProcessScheduler")
        worker = TFProcessScheduler(
            ps_num=args.ps_num,
            worker_num=args.worker_num,
            evaluator_num=args.evaluator_num,
            conf=args.conf,
            parsed_args=args,
        )
        # to do use constants

        if PlatformConstants.Ray() in platform:
            worker.set_start_subprocess(mock_ray_platform_subprocess)
        elif PlatformConstants.Kubernetes() in platform:
            worker.set_start_subprocess(mock_k8s_platform_subprocess)
        else:
            detail_trace_back = traceback.format_exc()
            logger.error(detail_trace_back)

    logger.info(
        "Running platform: %s, worker action: %s",
        args.platform,
        args.worker_action,
    )
    if args.worker_action == PlatformConstants.WorkerActionRun():
        return worker.run()


def run(other_arguments=None):
    """Entrance
    Args:
        other_arguments: dict of complex arguments
    """
    args = get_parsed_args()
    return execute(args)
