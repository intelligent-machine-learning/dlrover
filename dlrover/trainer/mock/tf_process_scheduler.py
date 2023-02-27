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

import copy
import itertools
import json
import os
import subprocess
import sys

from dlrover.trainer.mock.base_process_scheduler import BaseProcessScheduler
from dlrover.trainer.util.log_util import default_logger as logger
from dlrover.trainer.util.net_util import get_available_port


def mock_ray_platform_subprocess(tf_config):
    """
    start process using subprocess
    """
    argv = sys.argv
    task_type = tf_config["task"]["type"]
    task_id = tf_config["task"]["index"]
    worker_argv = [sys.executable, "-m", "dlrover.trainer.entry.local_entry"]

    worker_argv.extend(argv[1:])
    worker_argv.extend(["--platform", "ray"])
    worker_argv.extend(["--mock", "True"])
    worker_argv.extend(["--task_type", task_type])
    worker_argv.extend(["--task_id", str(task_id)])

    logger.info(worker_argv)
    env = dict(os.environ)
    fild_path = os.path.dirname(__file__)
    fild_path = fild_path.split("/")
    python_path = "/".join(fild_path[: len(fild_path) - 3])
    logger.info(python_path)
    env.update(
        {
            "PYTHONPATH": python_path,
        }
    )

    logger.info(json.dumps(tf_config))
    env["WORKFLOW_ID"] = os.getenv("WORKFLOW_ID", default="test_id")
    env["USERNUMBER"] = os.getenv("USERNUMBER", default="test_user")
    process = subprocess.Popen(worker_argv, shell=False, env=env)
    return process


def mock_k8s_platform_subprocess(tf_config):
    """
    start process using subprocess
    """
    argv = sys.argv
    worker_argv = [sys.executable, "-m", "dlrover.trainer.entry.local_entry"]
    worker_argv.extend(argv[1:])
    worker_argv.extend(["--platform", "kubernetes"])
    logger.info(worker_argv)
    env = dict(os.environ)
    fild_path = os.path.dirname(__file__)
    fild_path = fild_path.split("/")
    python_path = "/".join(fild_path[: len(fild_path) - 3])
    logger.info(python_path)
    env.update(
        {
            "TF_CONFIG": json.dumps(tf_config),
            "PYTHONPATH": python_path,
        }
    )
    logger.info(json.dumps(tf_config))
    env["WORKFLOW_ID"] = os.getenv("WORKFLOW_ID", default="test_id")
    env["USERNUMBER"] = os.getenv("USERNUMBER", default="test_user")
    process = subprocess.Popen(worker_argv, shell=False, env=env)
    return process


class TFProcessScheduler(BaseProcessScheduler):
    def __init__(
        self,
        ps_num=None,
        worker_num=None,
        evaluator_num=None,
        conf=None,
        parsed_args=None,
    ):
        super(TFProcessScheduler, self).__init__()
        assert worker_num >= 1, "worker number should be as least 1"
        assert evaluator_num <= 1, "evaluator number should be as most 1"
        self.ps_num = ps_num
        self.chief_num = 1
        self.worker_num = worker_num - 1
        self.evaluator_num = evaluator_num
        self.start_subprocess = None

    def set_start_subprocess(self, start_subprocess):
        self.start_subprocess = start_subprocess

    def prepare_cluster(self):
        """
        get available ports for ps/worker/evaluator
        """
        ps_ports = [get_available_port() for i in range(self.ps_num)]
        worker_ports = [get_available_port() for i in range(self.worker_num)]
        chief_ports = [get_available_port() for i in range(self.chief_num)]
        evaluator_ports = [
            get_available_port() for i in range(self.evaluator_num)
        ]
        cluster_info = {}
        if self.worker_num > 0:
            cluster_info.update({"worker": worker_ports})
        if self.ps_num > 0:
            cluster_info.update({"ps": ps_ports})
        if self.chief_num == 1:
            cluster_info.update({"chief": chief_ports})
        if self.evaluator_num == 1:
            cluster_info.update({"evaluator": evaluator_ports})
        self.tf_cluster_spec = {"cluster": cluster_info}

    def update_spec_and_start_process(self, task_spec):
        tf_cluster_spec = copy.deepcopy(self.tf_cluster_spec)
        tf_cluster_spec.update(task_spec)
        p = self.start_subprocess(tf_cluster_spec)
        return p

    def start_chief_process(self):
        chief_task_spec = {
            "task": {
                "type": "chief",
                "index": 0,
            }
        }
        chief_process = []
        p = self.update_spec_and_start_process(chief_task_spec)
        chief_process.append(p)
        return chief_process

    def start_worker_process(self):
        worker_process = []
        for i in range(self.worker_num):
            worker_task_spec = {
                "task": {
                    "type": "worker",
                    "index": i,
                }
            }
            p = self.update_spec_and_start_process(worker_task_spec)
            worker_process.append(p)
        return worker_process

    def start_ps_process(self):
        ps_process = []
        for i in range(self.ps_num):
            ps_task_spec = {
                "task": {
                    "type": "ps",
                    "index": i,
                }
            }
            p = self.update_spec_and_start_process(ps_task_spec)
            ps_process.append(p)
        return ps_process

    def start_evaluator_process(self):
        evaluator_process = []
        for i in range(self.evaluator_num):
            evaluator_task_spec = {
                "task": {
                    "type": "evaluator",
                    "index": i,
                }
            }
            p = self.update_spec_and_start_process(evaluator_task_spec)
            evaluator_process.append(p)
        return evaluator_process

    def run_process(self):
        self.prepare_cluster()
        ps_process = self.start_ps_process()
        chief_process = self.start_chief_process()
        evaluator_process = self.start_evaluator_process()
        worker_process = self.start_worker_process()
        self.all_processes = {
            "chief_process": chief_process,
            "ps_process": ps_process,
            "worker_process": worker_process,
            "evaluator_process": evaluator_process,
        }
        all_process = list(itertools.chain(*self.all_processes.values()))
        all_process_pid = [p.pid for p in all_process]
        logger.info("all processes are {}".format(all_process_pid))
        self.waiting_process = chief_process + worker_process
