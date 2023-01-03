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

from dlrover.trainer.tensorflow.executor.estimator_executor import (
    EstimatorExecutor,
)
from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.util.conf_util import get_conf
from dlrover.trainer.util.log_util import default_logger as logger


class TFKubernetesWorker:
    """TFKubemakerWorker"""

    def __init__(self, args):
        """
        Argument:
            args: result of parsed command line arguments
        """
        self._args = args
        task_conf = get_conf(py_conf=args.conf)
        self.init_executor(task_conf)

    def init_executor(self, task_conf):
        self.estimator = EstimatorExecutor(task_conf)

    def start_failover_monitor(self):
        pass

    def run(self):
        global_dict = common_util.GlobalDict()
        global_dict["executor"] = self.estimator
        self.start_failover_monitor()
        logger.info("KubernetesWorker is running!")
        self.estimator.start_server()
        if self.estimator.task_type == "ps":
            logger.info("ps server join")
            self.estimator.server.join()
        else:
            self.estimator.train_and_evaluate()
