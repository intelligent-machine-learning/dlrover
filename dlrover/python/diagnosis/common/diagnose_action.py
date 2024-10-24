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
import threading
from datetime import datetime
from dlrover.python.common.time import has_expired
from typing import List
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisAction as DiagnosisActionConstants,
)


class DiagnoseAction:
    def __init__(self, timestamp=0, expired_time_period=0, action="", rank=DiagnosisConstant.ANY_RANK):
        self.action = action
        if timestamp == 0:
            self.timestamp = int(round(datetime.now().timestamp()))
        else:
            self.timestamp= timestamp

        if expired_time_period == 0:
            self.expired_time_period = DiagnosisActionConstants.ACTION_EXPIRED_TIME_PERIOD
        else:
            self.expired_time_period = expired_time_period
        # rank indicates the rank (worker) which is to
        # execute this action. Rank = -1 indicates the
        # master is to execute the action
        self.rank = rank

    def set_action(self, action: str):
        self.action = action

    def has_expired(self) -> bool:
        return has_expired(self.timestamp, self.expired_time_period)


def is_same_action(action1: DiagnoseAction, action2: DiagnoseAction) -> bool:
    return action1.action == action2.action and action1.rank == action2.rank


class DiagnoseActionQueue:
    def __init__(self):
        self._actions: List[DiagnoseAction] = []
        self._lock = threading.Lock()

    def add_action(self, new_action: DiagnoseAction):
        with self._lock:
            for action in self._actions:
                if is_same_action(new_action, action):
                    return
            logger.info(f"enqueue action {new_action.action} of {new_action.rank}")
            self._actions.append(new_action)

    def _remove_expired_actions(self):
        with self._lock:
            actions = []
            for action in self._actions:
                if not action.has_expired():
                    actions.append(action)
                else:
                    logger.info(f"Action {action} has expired")

            self._actions = actions

    def next_actions(self, rank=DiagnosisConstant.ANY_RANK) -> List[DiagnoseAction]:
        self._remove_expired_actions()
        with self._lock:
            rank_actions = []
            remain_actions = []
            for action in self._actions:
                if action.rank == rank or rank == DiagnosisConstant.ANY_RANK:
                    rank_actions.append(action)
                else:
                    remain_actions.append(action)
            return rank_actions


