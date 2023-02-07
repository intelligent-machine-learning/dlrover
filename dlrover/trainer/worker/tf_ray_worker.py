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

import argparse
import os
import threading

from dlrover.python.elastic_agent.master_client import GlobalMasterClient
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

master_client = GlobalMasterClient.MASTER_CLIENT


class TFRayWorker:
    """TFRayWorker"""

    def __init__(self, args):
        """
        Argument:
            args: result of parsed command line arguments
        """
        self._args = self.transform_args_to_dict(args)
        logger.info(args)
        task_conf = get_conf(py_conf=self._args.get("conf"))
        self._task_conf = task_conf
        self.init_executor(task_conf)
        # self.run()
        if self._args.get("platform") == "ray":
            self.init_and_train()

    def transform_args_to_dict(self, args):
        if isinstance(args, argparse.Namespace):
            args = args.__dict__
        return args

    def parse_worker_type_and_id(self):
        task_id, task_type = self._args.get("task_id"), self._args.get(
            "task_type"
        )
        return task_id, task_type

    def init_and_train(self):
        """
        ray remote 调用时同步等待，通过线程，异步启动训练线程
        """
        t = threading.Thread(target=self.run)
        t.setDaemon(True)
        t.start()

    def init_executor(self, task_conf):
        self.estimator = EstimatorExecutor(task_conf)

    def start_failover_monitor(self):
        if self._args.get("enable_auto_scaling", None):
            self._task_conf.put(TFConstants.EnableDynamicSharding.name, True)
            self.tensorflow_failover = TensorflowFailover()
            self.tensorflow_failover.start_failover_monitor()

    def get_ps_cluster(self):

        if self._args.get("mock"):
            while True:
                ps_num = 0
                ps_cluster = []
                dir_list = os.listdir("./")
                for file in dir_list:
                    if file.startswith("ps_address_"):
                        ps_num += 1
                        address = file.split("_")[-1]
                        ps_cluster.append(address)
                if ps_num == self._args.get("ps_num"):
                    break
        else:
            while True:
                ps_nodes, _ = master_client.query_ps_nodes()
                ps_cluster = [i.addr for i in ps_nodes if i.addr != ""]
                if len(ps_cluster) == self._args.get("ps_num"):
                    break
        return ps_cluster

    def report_ps_address(self, address):
        if self._args.get("mock"):
            file_name = "ps_address_{}".format(address)
            with open(file_name, "w") as f:
                f.write("")
        else:
            master_client.update_node_addr(
                self.task_type, self.task_id, address
            )

    def run(self):
        global_dict = common_util.GlobalDict()
        global_dict["executor"] = self.estimator
        # self.start_failover_monitor()
        logger.info("RayWorker is running!")
        self.estimator.start_server()
        address = self.estimator.address
        task_id, task_type = self.parse_worker_type_and_id()
        self.task_id, self.task_type = task_id, task_type
        self.estimator.task_type = task_type

        if task_type != "ps":
            ps_cluster = self.get_ps_cluster()
            tf_config = {
                "cluster": {"ps": ps_cluster, task_type: [address]},
                "task": {"type": task_type, "index": task_id},
            }
            self.estimator.set_tf_config(tf_config)
        # upload server address
        # get_current_server address
        if self.estimator.task_type == TFConstants.PS():
            self.report_ps_address(address)
            logger.info("ps server join")
            self.estimator.server.join()
        else:
            self.estimator.train_and_evaluate()

    def health_check(self):
        return "OK"
