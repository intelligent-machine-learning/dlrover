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

import time

from dlrover.python.elastic_agent.tensorflow.elastic_ps import ElasticPsClient
from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.util.log_util import default_logger as logger


class FailoverClient:
    """
    FailoverClient interacts with dlrover master.
    It get's ps address, ps/worker global/local version.
    """

    def __init__(self, role=None):
        logger.info("initiating FailoverClient")
        self.role = role
        task_type, task_id = role.split(":")
        task_id = int(task_id)
        self.task_type = task_type
        self.task_id = task_id
        if self.task_type == TFConstants.Worker():
            self.task_id = +1
        if self.task_type == TFConstants.Chief():
            self.task_type = "worker"
        self._client = ElasticPsClient(self.task_type, self.task_id)
        logger.info(
            "ElasticPsService is created, task_type: {} and task_id {}.".format(  # noqa : E501
                task_type, task_id
            )
        )
        self.ps_client = ElasticPsClient(TFConstants.PS(), 0)

    def get_local_version(self):
        local_version = self._client.get_local_cluster_version()
        logger.info("get local version : %s.", local_version)
        return local_version

    def get_global_version(self):
        global_version = self.ps_client.get_global_cluster_version()
        logger.info("get ps global version : %s.", global_version)
        return global_version

    def ready_for_ps_relaunch(self):
        logger.info("Noticing dlrover master that it's ready for ps relaunch")
        self._client.ready_for_ps_relaunch()

    def set_global_version(self, version=0):
        self.ps_client.update_global_cluster_version(version)
        logger.info("successfully set ps global version: %s.", version)

    def set_local_version(self, version=0):
        self._client.update_local_cluster_version(version)
        logger.info("successfully set local version: %s.", version)

    def get_training_ps_addr(self):
        ps_nodes, _ = self._client.get_all_ps_nodes()
        return [n.addr for n in ps_nodes]

    def init_version(self, version=0):
        logger.info("initiating local and global version")
        local_version = self.get_local_version()
        global_version = self.get_global_version()
        if local_version == 0 and self.task_type == TFConstants.PS():
            version = local_version + 1
            self.set_local_version(version)
            if self.task_id == 0:
                # ps:0 updates global version while
                # other ps waiting for global version to be updated
                self.set_global_version(version)
            else:
                while global_version == 0:
                    global_version = self.get_global_version()
                    time.sleep(3)
                    logger.info(
                        "Waiting for ps:0 updating global version from 0 to 1."
                    )

        if (
            self.task_type in [TFConstants.Worker(), TFConstants.Chief()]
            and local_version == 0
        ):
            self.set_local_version(1)
            while global_version == 0:
                # workers waits for global version to be updated to 1
                global_version = self.get_global_version()
                time.sleep(3)
                logger.info(
                    "Waiting for ps-0 updating global version from 0 to 1."
                )
            version = self.get_local_version()
            logger.info(
                "{}:{} local version is {}".format(
                    self.task_type, self.task_id, version
                )
            )
