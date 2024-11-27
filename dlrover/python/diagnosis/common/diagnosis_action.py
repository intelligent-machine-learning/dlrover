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
import queue
import threading
from abc import ABCMeta
from datetime import datetime
from queue import Queue
from typing import Dict, Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.util.time_util import has_expired


class DiagnosisAction(metaclass=ABCMeta):
    """
    Operation to be done after diagnostician.

    Args:
        action_type (DiagnosisActionType): Type of action.
        instance (Optional): Instance id(nod id). Defaults to -1(Master).
        timestamp (Optional[datetime.datetime]): Timestamp of action. Unit: ms.
            Defaults to current time.
        expired_time_period (Optional): Milliseconds of expired time period.
            Unit: ms. Defaults to 60 seconds.
    """

    def __init__(
        self,
        action_type=DiagnosisActionType.NONE,
        instance: int = DiagnosisConstant.LOCAL_INSTANCE,
        timestamp: int = 0,
        expired_time_period: int = 60 * 1000,
    ):
        self._action_type = action_type
        self._instance: int = instance
        if timestamp == 0:
            self._timestamp = int(datetime.now().timestamp())
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

    def is_needed(self):
        return (
            not self.is_expired()
            and self._action_type != DiagnosisActionType.NONE
        )

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))

    def __repr__(self):
        return (
            f"action_type:{self._action_type};"
            f"instance:{self._instance};"
            f"timestamp:{self._timestamp};"
            f"expired_time_period:{self._expired_time_period}"
        )


class NoAction(DiagnosisAction):
    def __init__(self):
        super(NoAction, self).__init__()


class EventAction(DiagnosisAction):
    """Output the specified event."""

    def __init__(
        self,
        event_type: str = "",
        event_instance: str = "",
        event_action: str = "",
        event_msg: str = "",
        event_labels: Optional[Dict[str, str]] = None,
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


class NodeAction(DiagnosisAction):
    def __init__(
        self,
        node_status: str = "",
        reason: str = "",
        node_id=DiagnosisConstant.LOCAL_INSTANCE,
        action_type=DiagnosisActionType.NONE,
        timestamp=0,
        expired_time_period=0,
    ):
        super().__init__(
            action_type,
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
        self._actions: Dict[int, Queue[DiagnosisAction]] = {}
        self._lock = threading.Lock()

    def add_action(self, new_action: DiagnosisAction):
        with self._lock:
            instance = new_action.instance
            if instance not in self._actions:
                self._actions[instance] = Queue(maxsize=10)
            ins_actions = self._actions[instance]
            try:
                if new_action.is_needed():
                    ins_actions.put(new_action, timeout=3)
                    logger.info(f"New diagnosis action {new_action}")
            except queue.Full:
                logger.warning(
                    f"Diagnosis actions for {instance} is full, "
                    f"skip action: {new_action}."
                )

    def next_action(
        self,
        instance=DiagnosisConstant.LOCAL_INSTANCE,
    ) -> DiagnosisAction:
        with self._lock:
            while True:
                if (
                    instance not in self._actions
                    or self._actions[instance].empty()
                ):
                    return DiagnosisAction()
                action = self._actions[instance].get()
                if not action.is_expired():
                    return action
                else:
                    logger.info(f"Skip expired diagnosis action: {action}.")
