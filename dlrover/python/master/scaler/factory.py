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

from dlrover.python.common.constants import PlatformType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.scaler.elasticjob_scaler import ElasticJobScaler
from dlrover.python.master.scaler.pod_scaler import PodScaler
from dlrover.python.master.scaler.ray_scaler import ActorScaler


def new_job_scaler(platform, job_name, namespace):
    logger.info("New %s JobScaler", platform)
    if platform == PlatformType.KUBERNETES:
        return ElasticJobScaler(job_name, namespace)
    elif platform == PlatformType.PY_KUBERNETES:
        return PodScaler(job_name, namespace)
    elif platform == PlatformType.RAY:
        return ActorScaler(job_name, namespace)
