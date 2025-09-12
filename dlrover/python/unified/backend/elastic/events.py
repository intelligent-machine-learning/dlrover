# Copyright 2025 The DLRover Authors. All rights reserved.
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

from dlrover.python.training_event.predefined.common import CommonPredefined
from dlrover.python.unified.common.workload_desc import ElasticWorkloadDesc


class _ElasticMasterEvents:
    def __init__(self):
        self.__emitter = CommonPredefined("DLROVER_elastic_master")

    def inited(self, spec: ElasticWorkloadDesc):
        self.__emitter.instant("#inited", {"config": spec})

    def checking_workers(self):
        return self.__emitter.duration("#checking_workers")

    def doing_setup_workloads(self):
        return self.__emitter.duration("#setup_workloads")

    def starting_elastic_job(self):
        return self.__emitter.duration("#starting_elastic_job")

    def restarting(self):
        return self.__emitter.duration("#restarting")


ElasticMasterEvents = _ElasticMasterEvents()


class _ElasticWorkerEvents:
    def __init__(self):
        self.__emitter = CommonPredefined("DLROVER_elastic_worker")

    def init_process_group(self):
        return self.__emitter.duration("#init_process_group")

    def comm_check(self):
        return self.__emitter.duration("#comm_check")


ElasticWorkerEvents = _ElasticWorkerEvents()
