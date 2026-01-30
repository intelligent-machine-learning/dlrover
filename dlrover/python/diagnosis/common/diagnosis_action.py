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
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

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
        executable_time_period (Optional): This action can only be executed
            after given time period. Unit: s. Defaults to 0.
    """

    def __init__(
        self,
        action_type: str = DiagnosisActionType.NONE,
        instance: int = DiagnosisConstant.LOCAL_INSTANCE,
        timestamp: int = 0,
        expired_time_period: int = 60 * 1000,
        executable_time_period: int = 0,
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

        self._executable_time_period = executable_time_period

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
    def expired_time_period(self):
        return self._expired_time_period

    @property
    def expired_timestamp(self):
        return self._timestamp + self._expired_time_period

    def is_expired(self) -> bool:
        return has_expired(self._timestamp, self._expired_time_period)

    def is_executable(self):
        return has_expired(self._timestamp, self._executable_time_period)

    def to_json(self):
        data = {k.lstrip("_"): v for k, v in self.__dict__.items()}
        return json.dumps(data)

    def update_timestamp(
        self,
        timestamp: float = 0,
        expired_time_period: int = 0,
        executable_time_period: int = 0,
    ):
        if timestamp > 0:
            self._timestamp = int(timestamp)
        if expired_time_period > 0:
            self._expired_time_period = expired_time_period
        if executable_time_period > 0:
            self._expired_time_period = expired_time_period

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
    def __init__(self, **kwargs):
        super().__init__()


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
        executable_time_period=0,
        instance=DiagnosisConstant.MASTER_INSTANCE,
        **kwargs,
    ):
        super().__init__(
            action_type=DiagnosisActionType.EVENT,
            instance=instance,
            timestamp=timestamp,
            expired_time_period=expired_time_period,
            executable_time_period=executable_time_period,
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

    def __repr__(self):
        return (
            f"action_type:{self._action_type};"
            f"instance:{self._instance};"
            f"timestamp:{self._timestamp};"
            f"expired_time_period:{self._expired_time_period};"
            f"event_type:{self._event_type};"
            f"event_instance:{self._event_type};"
            f"event_action:{self._event_instance};"
            f"event_msg:{self._event_msg};"
            f"event_labels:{self._event_labels}"
        )


class NodeAction(DiagnosisAction):
    def __init__(
        self,
        node_id: int,
        node_type: str,
        node_status: str = "",
        reason: str = "",
        instance=DiagnosisConstant.MASTER_INSTANCE,
        action_type=DiagnosisActionType.NONE,
        timestamp=0,
        expired_time_period=0,
        **kwargs,
    ):
        super().__init__(
            action_type,
            instance,
            timestamp,
            expired_time_period,
        )
        self._node_id = node_id
        self._node_type = node_type
        self._node_status = node_status
        self._reason = reason

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type(self):
        return self._node_type

    @property
    def node_status(self):
        return self._node_status

    @property
    def reason(self):
        return self._reason

    def __repr__(self):
        return (
            f"action_type:{self._action_type};"
            f"instance:{self._instance};"
            f"timestamp:{self._timestamp};"
            f"expired_time_period:{self._expired_time_period};"
            f"node_id:{self._node_id};"
            f"node_type:{self._node_type};"
            f"node_status:{self._node_status};"
            f"reason:{self._reason}"
        )


class JobAction(DiagnosisAction):
    def __init__(
        self,
        action_type: str,
        reason: str = "",
        msg: str = "",
        **kwargs,
    ):
        super().__init__(
            action_type,
            DiagnosisConstant.MASTER_INSTANCE,
            0,
            0,
        )
        self._reason = reason
        self._msg = msg

    @property
    def reason(self):
        return self._reason

    @property
    def msg(self):
        return self._msg

    def __repr__(self):
        return (
            f"action_type:{self._action_type};"
            f"instance:{self._instance};"
            f"timestamp:{self._timestamp};"
            f"expired_time_period:{self._expired_time_period};"
            f"reason:{self._reason};"
            f"msg:{self._msg}"
        )


class JobAbortionAction(JobAction):
    def __init__(
        self,
        reason: str = "",
        msg: str = "",
        **kwargs,
    ):
        super().__init__(
            action_type=DiagnosisActionType.JOB_ABORT,
            reason=reason,
            msg=msg,
        )


class JobRestartAction(JobAction):
    def __init__(
        self,
        reason: str = "",
        msg: str = "",
        **kwargs,
    ):
        super().__init__(
            action_type=DiagnosisActionType.JOB_RESTART,
            reason=reason,
            msg=msg,
        )


def is_same_action(action1: DiagnosisAction, action2: DiagnosisAction) -> bool:
    if isinstance(action1, EventAction) and isinstance(action2, EventAction):
        action1.__class__ = EventAction
        action2.__class__ = EventAction
        if (
            action1.event_type == action2.event_type
            and action1.instance == action2.instance
            and action1.action_type == action2.action_type
            and action1.event_msg == action2.event_msg
            and not action1.is_expired()
            and not action2.is_expired()
        ):
            return True
    return False


class DiagnosisActionQueue:
    def __init__(self):
        self._actions: Dict[int, deque[DiagnosisAction]] = {}
        self._lock = threading.Lock()

    def add_action(self, new_action: DiagnosisAction):
        with self._lock:
            instance = new_action.instance
            if instance not in self._actions:
                self._actions[instance] = deque(
                    maxlen=DiagnosisConstant.MAX_ACTION_QUEUE_SIZE
                )
            actions = self._actions[instance]
            try:
                for action in actions:
                    if (
                        is_same_action(new_action, action)
                        or not action.is_needed()
                    ):
                        return
                actions.append(new_action)
                logger.info(f"New diagnosis action {new_action}")
            except Exception as e:
                logger.warning(f"Add action errors: {e}.")

    def clear(self):
        with self._lock:
            self._actions.clear()

    def len(self):
        with self._lock:
            return sum(len(d) for d in self._actions.values())

    def next_action(
        self,
        instance=DiagnosisConstant.LOCAL_INSTANCE,
    ) -> DiagnosisAction:
        with self._lock:
            if (
                instance not in self._actions
                or len(self._actions[instance]) == 0
            ):
                return NoAction()

            logger.debug(f"Dump {instance} actions before: {self._actions}")
            actions = self._actions[instance]

            waiting_actions: Deque[DiagnosisAction] = deque()
            while len(actions) > 0:
                action = actions.popleft()
                if action.is_expired():
                    logger.info(
                        f"Skip expired diagnosis action({instance}): {action}"
                    )
                elif action.is_executable():
                    while len(waiting_actions) > 0:
                        waiting_action = waiting_actions.pop()
                        actions.appendleft(waiting_action)
                    logger.info(
                        f"Get diagnosis action({instance}): {action}, "
                        f"Remain: {actions}/{len(actions)}"
                    )
                    return action
                else:
                    waiting_actions.append(action)

            self._actions[instance] = waiting_actions
            logger.debug(f"Dump {instance} actions after: {self._actions}")
            return NoAction()
