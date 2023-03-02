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
import threading

from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.executor.estimator_executor import (
    EstimatorExecutor,
)
from dlrover.trainer.tensorflow.failover.tensorflow_failover import (
    TensorflowFailover,
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
        self._task_conf = task_conf
        self.estimator_server_started = False
        self.init_executor(task_conf)

    def init_executor(self, task_conf):
        logger.info("init_executor")
        self.estimator = EstimatorExecutor(task_conf)

    def start_failover_monitor(self):
        if self._args.enable_auto_scaling:
            self._task_conf.put(TFConstants.EnableDynamicSharding.name, True)
            self.tensorflow_failover = TensorflowFailover()
            self.tensorflow_failover.start_failover_monitor()

    def run_ps(self):
        logger.info("ps server join")
        self.estimator.server.join()

    def run_worker(self):
        self.estimator.train_and_evaluate()

    def run(self):
        logger.info("KubernetesWorker is running!")
        while True:
            global_dict = common_util.GlobalDict()
            self.start_failover_monitor()
            global_dict["executor"] = self.estimator
            self.estimator.prepare()
            if not self.estimator_server_started:
                self.estimator.start_server()
                self.estimator_server_started = True
            if self.estimator.task_type == TFConstants.PS():
                run_thread = threading.Thread(target=self.run_ps)
            else:
                run_thread = threading.Thread(target=self.run_worker)
            run_thread.start()
            run_thread.join()
            if not run_thread.is_alive() and global_dict.get(
                TFConstants.RelaunchForPs.name, TFConstants.RelaunchForPs()
            ):
                logger.info("ps is migrating or scaling")
                global_dict.clear()
                self.init_executor(self._task_conf)
                continue
            else:
                break
