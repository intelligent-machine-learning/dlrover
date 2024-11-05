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
from abc import ABCMeta
from datetime import datetime
from typing import Dict, List

from dlrover.python.common.constants import NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionConstant,
    DiagnosisConstant,
)
from dlrover.python.util.time_util import has_expired


class DiagnosisAction(metaclass=ABCMeta):
    def __init__(
        self,
        action_type: str,
        instance: int,
        timestamp=0,
        expired_time_period=0,
    ):
        self.action_type = action_type
        self.instance = instance
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp = timestamp

        if expired_time_period == 0:
            self.expired_time_period = (
                DiagnosisActionConstant.ACTION_EXPIRED_TIME_PERIOD
            )
        else:
            self.expired_time_period = expired_time_period

    def has_expired(self) -> bool:
        return has_expired(self.timestamp, self.expired_time_period)

    def to_json(self):
        data = {k.lstrip("_"): v for k, v in self.__dict__.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))


class DiagnosisNodeAction(DiagnosisAction):
    def __init__(
        self,
        timestamp=0,
        expired_time_period=0,
        action="",
        node_type=NodeType.WORKER,
        instance=DiagnosisConstant.LOCAL_INSTANCE,
    ):
        super().__init__(
            DiagnosisActionConstant.TYPE_NODE,
            instance,
            timestamp,
            expired_time_period,
        )
        self.action = action
        self.node_type = node_type

    def update_action(self, action: str):
        self.action = action


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
            logger.info(f"enqueue action {new_action}")
            ins_actions.append(new_action)

    def _remove_expired_actions(self):
        with self._lock:
            for instance in self._actions.keys():
                action_queue = self._actions[instance]
                actions = []
                for action in action_queue:
                    if not action.has_expired():
                        actions.append(action)
                    else:
                        logger.info(f"Action {action} has expired")
                self._actions[instance] = actions

    def next_actions(
        self,
        instance=DiagnosisConstant.LOCAL_INSTANCE,
        action_type=DiagnosisActionConstant.ACTION_TYPE_ANY,
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
                    action_type == DiagnosisActionConstant.TYPE_NODE
                    or action_type == action.action_type
                ):
                    deque_actions.append(action)
                else:
                    remain_actions.append(action)
            self._actions[instance] = remain_actions
            return deque_actions
