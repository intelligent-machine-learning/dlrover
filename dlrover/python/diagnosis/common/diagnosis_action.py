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

import json
import threading
import time
from abc import ABCMeta
from typing import Dict, List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.util.time_util import has_expired


class DiagnosisAction(metaclass=ABCMeta):
    def __init__(
        self,
        action_type=DiagnosisActionType.NONE,
        instance=DiagnosisConstant.MASTER,
        timestamp=0,
        expired_time_period=0,
    ):
        self._action_type = action_type
        self._instance = instance
        if timestamp == 0:
            self._timestamp = time.time()
        else:
            self._timestamp = timestamp

        if expired_time_period == 0:
            self._expired_time_period = (
                DiagnosisConstant.ACTION_EXPIRED_TIME_PERIOD_DEFAULT
            )
        else:
            self._expired_time_period = expired_time_period

    @property
    def action_type(self):
        return self._action_type

    @property
    def instance(self):
        return self._instance

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def expired_timestamp(self):
        return self._timestamp + self._expired_time_period

    def is_expired(self) -> bool:
        return has_expired(self._timestamp, self._expired_time_period)

    def to_json(self):
        data = {k.lstrip("_"): v for k, v in self.__dict__.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))


class EventAction(DiagnosisAction):
    """Output the specified event."""

    def __init__(
        self,
        event_type: str = "",
        event_instance: str = "",
        event_action: str = "",
        event_msg: str = "",
        event_labels: Dict[str, str] = {},
        timestamp=0,
        expired_time_period=0,
    ):
        super().__init__(
            DiagnosisActionType.EVENT,
            timestamp=timestamp,
            expired_time_period=expired_time_period,
        )
        self._event_type = event_type
        self._event_instance = event_instance
        self._event_action = event_action
        self._event_msg = event_msg
        self._event_labels = event_labels

    @property
    def event_type(self):
        return self._event_type

    @property
    def event_instance(self):
        return self._event_instance

    @property
    def event_action(self):
        return self._event_action

    @property
    def event_msg(self):
        return self._event_msg

    @property
    def event_labels(self):
        return self._event_labels


class NodeRelaunchAction(DiagnosisAction):
    def __init__(
        self,
        node_id,
        node_status,
        reason,
        timestamp=0,
        expired_time_period=0,
    ):
        super().__init__(
            DiagnosisActionType.MASTER_RELAUNCH_WORKER,
            node_id,
            timestamp,
            expired_time_period,
        )
        self._node_status = node_status
        self._reason = reason

    @property
    def node_id(self):
        return self.instance

    @property
    def node_status(self):
        return self._node_status

    @property
    def reason(self):
        return self._reason


def is_same_action(action1: DiagnosisAction, action2: DiagnosisAction) -> bool:
    return False


class DiagnosisActionQueue:
    def __init__(self):
        self._actions: Dict[int, List[DiagnosisAction]] = {}
        self._lock = threading.Lock()

    def add_action(self, new_action: DiagnosisAction):
        with self._lock:
            instance = new_action.instance
            if instance not in self._actions:
                self._actions[instance] = []
            ins_actions = self._actions[instance]
            for action in ins_actions:
                if is_same_action(new_action, action):
                    return
            logger.info(f"New diagnosis action {new_action}")
            ins_actions.append(new_action)

    def _remove_expired_actions(self):
        with self._lock:
            for instance in self._actions.keys():
                action_queue = self._actions[instance]
                actions = []
                for action in action_queue:
                    if not action.is_expired():
                        actions.append(action)
                    else:
                        logger.info(f"Action {action} has expired")
                self._actions[instance] = actions

    def next_actions(
        self,
        instance=DiagnosisConstant.LOCAL_INSTANCE,
        action_type=DiagnosisActionType.ANY,
    ) -> List[DiagnosisAction]:
        self._remove_expired_actions()
        with self._lock:
            if instance not in self._actions:
                return []
            deque_actions = []
            remain_actions = []
            actions = self._actions[instance]
            for action in actions:
                if (
                    action_type == DiagnosisActionType.NODE_ACT
                    or action_type == action.action_type
                ):
                    deque_actions.append(action)
                else:
                    remain_actions.append(action)
            self._actions[instance] = remain_actions
            return deque_actions
