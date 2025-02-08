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

from dlrover.python.training_event.emitter import Process


class CommonEventName:
    WARNING = "#warning"
    EXCEPTION = "#exception"


class WarningType:
    DEPRECATED = "deprecated"


class CommonPredefined(Process):
    """
    Common predefined events.
    """

    def __init__(self, target: str):
        super().__init__(target)

    def warning(self, warning_type: WarningType, msg: str, **kwargs):
        """
        Emit a warning event, the warning event will be notified to the user.
        """
        self.instant(
            CommonEventName.WARNING,
            {
                "warning_type": warning_type,
                "msg": msg,
                **kwargs,
            },
        )
