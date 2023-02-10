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
from dlrover.python.master.watcher.k8s_watcher import (
    K8sScalePlanWatcher,
    PodWatcher,
)
from dlrover.python.master.watcher.ray_watcher import (
    ActorWatcher,
    RayScalePlanWatcher,
)


def new_node_watcher(platform, job_name, namespace):
    logger.info("New %s NodeWatcher", platform)
    if platform in (PlatformType.KUBERNETES, PlatformType.PY_KUBERNETES):
        return PodWatcher(job_name, namespace)
    elif platform in (PlatformType.RAY):
        return ActorWatcher(job_name, namespace)
    else:
        raise ValueError("Not support engine %s", platform)


def new_scale_plan_watcher(platform, job_name, namespace, job_uuid):
    logger.info("New %s NodeWatcher", platform)
    if platform in (PlatformType.KUBERNETES, PlatformType.PY_KUBERNETES):
        return K8sScalePlanWatcher(job_name, namespace, job_uuid)
    elif platform in (PlatformType.RAY):
        return RayScalePlanWatcher(job_name, namespace, job_uuid)
    else:
        raise ValueError("Not support engine %s", platform)
