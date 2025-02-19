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

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from dlrover.python.training_event.config import Config


class EventTargetName(object):
    TRAINER = "AtorchTrainerV2"
    SAVER = "AsyncSaver"


class EventTypeName(object):
    BEGIN = "BEGIN"
    END = "END"
    INSTANT = "INSTANT"


class EventType(Enum):
    """
    The type of the event.

    BEGIN: The beginning of a duration event.
    END: The end of a duration event.
    INSTANT: The instant of the event.

    duration event is a pair of BEGIN and END events, they are used
    to indicate a span of time. the event id is the same for the pair.

    instant event is a single event, indicating a point in time.
    """

    BEGIN = auto()
    END = auto()
    INSTANT = auto()


def _default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return f"<<non-serializable: {type(o).__qualname__}>>"


@dataclass(frozen=True)
class Event:
    """
    The event class.
    """

    pid: int
    event_id: str
    event_time: datetime
    name: str
    event_type: EventType
    content: dict
    target: str = ""
    max_event_prefix = 6

    def __str__(self):
        return "[%s] [%s] [%s] [%s] [%s] [%s] %s" % (
            self.event_time.isoformat(),
            self.pid,
            self.event_id,
            self.target,
            self.name,
            self.event_type.name,
            json.dumps(self.content, ensure_ascii=False, default=_default),
        )

    def to_dict(self):
        return {
            "pid": self.pid,
            "event_id": self.event_id,
            "event_time": self.event_time.isoformat(),
            "target": self.target,
            "name": self.name,
            "event_type": self.event_type.name,
            "content": self.content,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, default=_default)

    @classmethod
    def instant(
        cls,
        event_id: str,
        target: str,
        name: str,
        content: Optional[dict] = None,
    ):
        return Event(
            event_id=event_id,
            event_time=datetime.now(),
            target=target,
            name=name,
            event_type=EventType.INSTANT,
            content=content or {},
            pid=Config.singleton_instance().pid,
        )

    @classmethod
    def begin(cls, event_id: str, target: str, name: str, content: dict):
        return Event(
            event_id=event_id,
            event_time=datetime.now(),
            target=target,
            name=name,
            event_type=EventType.BEGIN,
            content=content,
            pid=Config.singleton_instance().pid,
        )

    @classmethod
    def end(cls, event_id: str, target: str, name: str, content: dict):
        return Event(
            event_id=event_id,
            event_time=datetime.now(),
            target=target,
            name=name,
            event_type=EventType.END,
            content=content,
            pid=Config.singleton_instance().pid,
        )
