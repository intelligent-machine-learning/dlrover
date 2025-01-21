import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class EventType(Enum):
    BEGIN = auto()
    END = auto()
    INSTANT = auto()


def _default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return f"<<non-serializable: {type(o).__qualname__}>>"


@dataclass(frozen=True)
class Event:
    event_time: datetime
    name: str
    event_type: EventType
    content: dict
    target: str = ""

    def __str__(self):
        return "[%s] [%s] [%s] [%s] %s" % (
            self.event_time.isoformat(),
            self.target,
            self.name,
            self.event_type.name,
            json.dumps(self.content, ensure_ascii=False, default=_default),
        )

    def to_dict(self):
        return {
            "event_time": self.event_time.isoformat(),
            "target": self.target,
            "name": self.name,
            "event_type": self.event_type.name,
            "content": self.content,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, default=_default)

    @classmethod
    def instant(cls, target: str, name: str, content: Optional[dict] = None):
        return Event(
            event_time=datetime.now(),
            target=target,
            name=name,
            event_type=EventType.INSTANT,
            content=content or {},
        )

    @classmethod
    def begin(cls, target: str, name: str, content: dict):
        return Event(
            event_time=datetime.now(),
            target=target,
            name=name,
            event_type=EventType.BEGIN,
            content=content,
        )

    @classmethod
    def end(cls, target: str, name: str, content: dict):
        return Event(
            event_time=datetime.now(),
            target=target,
            name=name,
            event_type=EventType.END,
            content=content,
        )
