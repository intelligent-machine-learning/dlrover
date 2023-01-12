#!/usr/bin/env python
# Copyright 2023 The DLRover Authors. All rights reserved.
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


# coding: utf-8

import json
import os
import threading
import time
from typing import List

from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.failover.failover_client import FailoverClient
from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.tensorflow.util.tf_patch_util import hotpatch_for_dynet
from dlrover.trainer.util.log_util import default_logger as logger


class TensorflowFailover:
    def __init__(self, role=None, failover_level=1):
        """
        Args:
            role: "ps:0", "worker:1"
            uuid_key: unique key that marks this cluster
            failover_level: switch for dynet
        """
        self._role = role
        logger.info(
            "initiating tensorflow_failover and failover level is {}".format(
                failover_level
            )
        )
        self._failover_level = failover_level
        hotpatch_for_dynet(failover_level)
        if common_util.should_failover(self._failover_level):
            self.init_for_dynet_from_tf_config()
        logger.info(
            "TensorflowFailover: role: %s, failover_level: %s,  ",
            role,
            failover_level,
        )

    def init_for_dynet_from_tf_config(self):

        tf_config = json.loads(os.environ.get("TF_CONFIG") or "{}")
        if not tf_config:
            logger.error(
                "TF_CONFIG should not be empty in distributed environment."
            )
            raise Exception(
                "TF_CONFIG should not be empty in distributed environment."
            )
        task_type = tf_config["task"]["type"]
        task_id = tf_config["task"]["index"]
        self._role = task_type + ":" + str(task_id)
        self._address = tf_config["cluster"][task_type][task_id]
        self.task_type, self.task_id = self._role.split(":")
        self.task_id = int(self.task_id)
        self._failover_client = FailoverClient(role=self._role)
        self._is_chief = (
            self.task_type == TFConstants.Chief.name and self.task_id == 0
        )
        self.prev_address = tf_config["cluster"]["ps"]
        self._failover_client.init_version()

    def start_failover_monitor(self):
        if self._role and TFConstants.Evaluator.name not in self._role:
            self._start_for_ps_migration()

    def _start_for_ps_migration(self):
        """
        Listening for ps address in the training cluster.
        If ps address changes, it means that dlrover master is
        going to migrate ps. When found ps address changes, it
        doesn't neede to restart worker and reconstuct graph.
        Only cluster info in session config and estimator.RunConfig
        needs to be refreshed. By building session with new config
        instead of restarting worker,
        """

        def monitor_fun():
            logger.info("Successfully to start monitor ps address!")
            while True:
                logger.info("Checking whether ps address changes")
                curr_address = self._failover_client.get_training_ps_addr()
                refresh_session = False
                logger.info(
                    "prev address is {} and current address is {}".format(
                        self.prev_address, curr_address
                    )
                )
                if "".join(curr_address) != "".join(self.prev_address):
                    self.prev_address = curr_address
                    refresh_session = True
                    logger.info("PS address changes, refresh session config")
                if refresh_session:
                    self.refresh_config(curr_address)
                time.sleep(10)

        self.runner = threading.Thread(target=monitor_fun)
        self.runner.start()

    def refresh_config(self, cluster_spec: List[str]):
        """Refresh session_creator._config
        Refresh the cluster information and rebuild a session
        linked to the new PS.

        Args:
         cluster_spec: list of `String`, which is the ps addresses
        """
        assert (
            len(cluster_spec) > 0
        ), "there should be as least one ps address in the training cluster"
        logger.info("ps cluster is {}".format(cluster_spec))
        global_dict = common_util.GlobalDict()
        session_creator = global_dict["session_creator"]
        config = session_creator._config
        if (
            config is not None
            and getattr(config, "cluster_def", None) is not None
        ):
            logger.info("before updating, session config is {}".format(config))
            # Get worker's index and address from previous session config
            # instead of TF_CONFIG
            for i in config.cluster_def.job:
                if i.name in [TFConstants.Worker, TFConstants.Chief]:
                    task = i.tasks[0]
                    if isinstance(task, dict):
                        ind, address = list(task.items())[0]
                    else:
                        ind = 0
                        address = task
            logger.info(
                "worker ind is {} and address is {}".format(ind, address)
            )
            len_job = len(config.cluster_def.job)
            for i in range(len_job):
                # pop ps and worker/chief info
                config.cluster_def.job.pop()
            worker_job = config.cluster_def.job.add()
            worker_job.name = self.task_type
            worker_job.tasks[ind] = address
            ps_job = config.cluster_def.job.add()
            ps_job.name = TFConstants.PS()
            for i, j in enumerate(cluster_spec):
                ps_job.tasks[i] = j
            logger.info(
                "after updating, session config is %s.", str(vars(config))
            )

            # TODO: before relaunch ps, there should a sync between all workers
            if self.task_type == "chief":
                self._failover_client.ready_for_ps_relaunch()
