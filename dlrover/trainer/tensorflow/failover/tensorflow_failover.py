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

from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.failover.failover_client import FailoverClient
from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.tensorflow.util.tf_env_util import (
    get_tf_config,
    get_tf_config_task_type_and_index,
)
from dlrover.trainer.util.log_util import default_logger as logger


class TensorflowFailover:
    def __init__(self, failover_client=FailoverClient, failover_level=1):
        """
        Args:
            role: "ps:0", "worker:1"
            uuid_key: unique key that marks this cluster
            failover_level: switch for dynet
        """
        logger.info(
            "initiating tensorflow_failover and failover level is {}".format(
                failover_level
            )
        )
        self._failover_level = failover_level
        if common_util.should_failover(self._failover_level):
            self.init_for_dynet()
            self._failover_client = failover_client(role=self._role)
            self._failover_client.init_version()

    def init_for_dynet(self):
        TF_CONFIG = get_tf_config()
        logger.info("TF_CONFIG is {}".format(TF_CONFIG))
        task_type, task_id = get_tf_config_task_type_and_index()
        self._role = task_type + ":" + str(task_id)
        if len(TF_CONFIG["cluster"][task_type]) > 1:
            self._address = TF_CONFIG["cluster"][task_type][task_id]
        else:
            self._address = TF_CONFIG["cluster"][task_type][0]
        if self._role is None:
            return
        self.task_type, self.task_id = self._role.split(":")
        self.task_id = int(self.task_id)
        self._is_chief = self.task_type == "chief" and self.task_id == 0
        self.curr_ps_address = TF_CONFIG["cluster"]["ps"]
        logger.info("Initial ps address is %s" % self.curr_ps_address)

    def start_failover_monitor(self):
        if self._role and self.task_type not in ["evaluator", "ps"]:
            self._start_failover_monitor()

    def _start_failover_monitor(self):
        def monitor_fun():
            logger.info("Successfully to start failover monitor!")
            while True:
                ps_address_changed, _ = self.ps_addresses_changed()
                if ps_address_changed:
                    self.refresh_env()
                    break
                time.sleep(10)

        self.runner = threading.Thread(target=monitor_fun)
        self.runner.setDaemon(True)
        self.runner.start()

    def ps_addresses_changed(self):
        """
        Check whether ps addresses changed.
        There are at least two kinds: 1) the num of ps
        addresses changed, 2) single ps address varies.
        """
        changed = False
        changed_type = None
        curr_address = self._failover_client.get_training_ps_addr()
        if "".join(curr_address) != "".join(self.curr_ps_address):
            if len(curr_address) != len(self.curr_ps_address):
                changed_type = "scaling"
            else:
                changed_type = "migrating"
            logger.info(
                "ps address changed from {} to {}.".format(
                    self.curr_ps_address, curr_address
                )
            )
            self.curr_ps_address = curr_address
            changed = True
        return changed, changed_type

    def refresh_env(self):
        """Refresh tf env
        update TF_CONFIG, when the training thread restarts
        estimator will use the new TF_CONFIG
        """
        global_dict = common_util.GlobalDict()
        global_dict["failover"] = self
        tf_config = get_tf_config()
        tf_config["cluster"]["ps"] = self.curr_ps_address
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        logger.info(
            "successfully refresh TF_CONFIFG %s" % os.environ["TF_CONFIG"]
        )

        if self._is_chief:
            # chief needs to do checkpoint and then
            # set global_dict[TFConstants.SaveCheckpoint.name] = True
            # In the checkpoint save hook,
            # chief set global_dict[TFConstants.RelaunchForPs.name] = True
            global_dict[TFConstants.SaveCheckpoint.name] = True
        else:
            # worker needs set
            # global_dict[TFConstants.RelaunchForPs.name] = True
            # and wait chief do checkpoints.
            global_dict[TFConstants.RelaunchForPs.name] = True
        logger.info("global dict is %s" % global_dict)
