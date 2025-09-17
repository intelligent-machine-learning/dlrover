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
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.enums import MasterStage


class _ControllerEvents:
    def __init__(self):
        self.__emitter = CommonPredefined("DLROVER_Controller")

    def inited(self, config: JobConfig):
        self.__emitter.instant("#inited", {"config": config})

    def stage_updated(self, old_stage: MasterStage, new_stage: MasterStage):
        self.__emitter.instant(
            "#stage_changed",
            {"old_stage": old_stage, "new_stage": new_stage},
        )

    def wait_ready(self, not_ready_count: int, total_count: int):
        self.__emitter.instant(
            "#wait_actors_ready",
            {"total_count": total_count, "not_ready_count": not_ready_count},
        )

    def node_check(self):
        return self.__emitter.duration("#node_check")

    def starting(self):
        return self.__emitter.duration("#starting")

    def stop_requested(self, reason: str, code: int):
        self.__emitter.instant(
            "#stop_requested", {"reason": reason, "code": code}
        )

    def stopping(self):
        return self.__emitter.duration("#stopping")

    def restarting(self):
        return self.__emitter.duration("#restarting")

    def saving(self):
        return self.__emitter.duration("#saving")

    def loading_state(self):
        return self.__emitter.duration("#loading_state")

    def failover_success(self):
        self.__emitter.instant("#failover_success")

    def failover_stop(self, stage: MasterStage):
        self.__emitter.instant("#failover_stop", {"stage": stage})

    def creating_pg(self):
        return self.__emitter.duration("#creating_placement_group")

    def creating_actors(self):
        return self.__emitter.duration("#creating_actors")

    def setup_actors(self):
        return self.__emitter.duration("#setup_actors")


ControllerEvents = _ControllerEvents()
