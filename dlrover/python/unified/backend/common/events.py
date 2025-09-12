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


class _BaseWorkerEvents:
    def __init__(self):
        self.__emitter = CommonPredefined("DLROVER_worker")

    def import_user_entrypoint(self, entrypoint: str):
        return self.__emitter.duration(
            "#import_user_entrypoint", {"entrypoint": entrypoint}
        )

    def instantiate_user_class(self, class_name: str):
        return self.__emitter.duration(
            "#instantiate_user_class", {"class_name": class_name}
        )

    def running(self):
        return self.__emitter.duration("#running")


BaseWorkerEvents = _BaseWorkerEvents()
