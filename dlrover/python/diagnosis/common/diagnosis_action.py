# Copyright 2024 The DLRover Authors. All rights reserved.
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

from typing import Dict, List

from dlrover.python.diagnosis.common.constants import DiagnosisActionType


class DiagnosisAction:
    """
    The action describes the expect operation after diagnostician.
    The action can be consumed by the master's job manager or directly used
    in training node.
    """

    def __init__(
        self,
        diagnosis_type: DiagnosisActionType = DiagnosisActionType.NO_ACTION,
        action_config={},
    ):
        """
        Args:
            diagnosis_type (DiagnosisActionType): The action type.
        """

        self._diagnosis_type = diagnosis_type
        self._action_config = action_config

    @property
    def diagnosis_type(self):
        return self._diagnosis_type

    @property
    def action_config(self):
        return self._action_config


class EventAction(DiagnosisAction):
    """Output the specified event."""

    def __init__(
        self,
        event_type: str = "",
        instance: str = "",
        action: str = "",
        msg: str = "",
        labels: Dict[str, str] = {},
    ):
        super().__init__(DiagnosisActionType.EVENT)
        self._event_type = event_type
        self._instance = instance
        self._action = action
        self._msg = msg
        self._labels = labels

    @property
    def event_type(self):
        return self._event_type

    @property
    def instance(self):
        return self._instance

    @property
    def action(self):
        return self._action

    @property
    def msg(self):
        return self._msg

    @property
    def labels(self):
        return self._labels


class NodeRelaunchAction(DiagnosisAction):
    """Relaunch the specified node."""

    def __init__(self, node_id, node_status, reason):
        super().__init__(DiagnosisActionType.MASTER_RELAUNCH_WORKER)
        self._node_id = node_id
        self._node_status = node_status
        self._reason = reason

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_status(self):
        return self._node_status

    @property
    def reason(self):
        return self._reason
