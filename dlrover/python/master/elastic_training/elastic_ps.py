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

from dlrover.python.common.constants import PSClusterVersionType
from dlrover.python.common.log import default_logger as logger


class ElasticPsService(object):
    def __init__(self):
        self._global_version = 0
        self._ps_local_version = {}
        self._worker_local_version = {}
        self._worker_restored_version = {}

    def inc_global_cluster_version(self):
        logger.info("Increment the global version to %s", self._global_version)
        self._global_version += 1

    def get_ps_version(self, version_type, ps_id):
        if version_type == PSClusterVersionType.GLOBAL:
            return self._global_version
        elif version_type == PSClusterVersionType.LOCAL:
            return self._ps_local_version.get(ps_id, 0)
        else:
            logger.warning("Unsupported type {}".format(version_type))
            return 0

    def update_ps_version(self, ps_id, version_type, version):
        if version_type == PSClusterVersionType.LOCAL:
            self._ps_local_version[ps_id] = version
            logger.info("PS local cluster version: %s", self._ps_local_version)
        elif version_type == PSClusterVersionType.GLOBAL:
            self._global_version = version
            logger.info("Global cluster version: %s", self._global_version)
        else:
            logger.warning("Unsupported type {}".format(version_type))

    def get_worker_version(self, version_type, worker_id):
        if version_type == PSClusterVersionType.GLOBAL:
            return self._global_version
        elif version_type == PSClusterVersionType.LOCAL:
            return self._worker_local_version.get(worker_id, 0)
        elif version_type == PSClusterVersionType.RESTORED:
            return self._worker_restored_version.get(worker_id, -1)
        else:
            logger.warning("Unsupported type {}".format(version_type))
            return 0

    def update_worker_version(self, worker_id, version_type, version):
        if version_type == PSClusterVersionType.LOCAL:
            self._worker_local_version[worker_id] = version
            logger.info(
                "Worker local cluster versions : {}".format(
                    self._worker_local_version
                )
            )
        elif version_type == PSClusterVersionType.RESTORED:
            self._worker_restored_version[worker_id] = version
            logger.info(
                "Worker restored cluster versions : {}".format(
                    self._worker_restored_version
                )
            )
        elif version_type == PSClusterVersionType.GLOBAL:
            logger.info(
                "Worker update global version from {} to {}".format(
                    self._global_version, version
                )
            )
            self._global_version = version
        else:
            logger.warning("Unsupported type {}".format(version_type))
