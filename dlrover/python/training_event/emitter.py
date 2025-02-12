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

import traceback
import uuid
from functools import wraps
from threading import Lock
from typing import Optional, Type

from dlrover.python.training_event.config import get_default_logger
from dlrover.python.training_event.event import Event
from dlrover.python.training_event.exporter import (
    EventExporter,
    get_default_exporter,
)

logger = get_default_logger()


def generate_event_id():
    """
    Generate a short unique id
    """
    return uuid.uuid4().hex[:8]


class EventEmitter:
    """
    EventEmitter is responsible to hold the target, and emit the event through
    the controller.
    """

    def __init__(self, target: str, exporter: EventExporter):
        self.target = target
        self.exporter = exporter

    def instant(self, name: str, content: Optional[dict] = None):
        if self.exporter is not None:
            event = Event.instant(
                generate_event_id(), self.target, name, content
            )
            self.exporter.export(event)

    def begin(self, event_id: str, name: str, content: dict):
        if self.exporter is not None:
            event = Event.begin(event_id, self.target, name, content)
            self.exporter.export(event)

    def end(self, event_id: str, name: str, content: dict):
        if self.exporter is not None:
            event = Event.end(event_id, self.target, name, content)
            self.exporter.export(event)


# singleton object for safe callable and chain call
class SafeCallable:
    """
    SafeCallable is a singleton stub object to avoid chain call error.
    """

    _lock = Lock()
    _instance: Optional["SafeCallable"] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SafeCallable, cls).__new__(cls)
        return cls._instance

    def __call__(self, *args, **kwargs):
        return SafeCallable()

    def __getattr__(self, item):
        return SafeCallable()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return


def safe_call(func):
    """Decorator: Safe handling of event-related calls"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.error(
                f"Failed in event processing {func.__name__}, exception: {e}, "
                f"traceback: {traceback_str}"
            )
            # return a safe callable object to avoid chain call error
            return SafeCallable()

    return wrapper


class SaveCallMeta(type):
    """Meta class: Automatically add safe call decorator to class methods"""

    def __new__(cls, name, bases, attrs):
        # add safe call decorator to all non-underscore-prefixed methods
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and not attr_name.startswith("_"):
                attrs[attr_name] = safe_call(attr_value)
        return super().__new__(cls, name, bases, attrs)


class SafeAttribute:
    """
    SafeAttribute is a class to avoid attribute missing error.
    """

    def __getattr__(self, item):
        logger.error(
            f"Attribute {item} from class {self.__class__.__name__} not found"
        )
        return SafeCallable()


class DurationSpan(SafeAttribute, metaclass=SaveCallMeta):
    """
    DurationSpan is a span of time, which can be used to emit a duration begin
    and end event.

    It support sub-span, extra args/dict, success/fail event and context
    management.
    User can inherit this class to add span-specific stages.
    """

    def __init__(
        self, emitter: EventEmitter, name: str, content: Optional[dict] = None
    ):
        self.emitter = emitter
        self.name = name
        self.content = content or {}
        self._is_begin = False
        self._is_end = False
        self._event_id = generate_event_id()

    def begin(self):
        """
        Begin the duration span, this will automatically call when use
        contextmanager.

        If the span is already begin, this will not emit anything.

        example:

        with duration_span:
            ...

        or

        duration_span.begin()

        Returns
        -------
        DurationSpan
            The same DurationSpan object for chain call.
        """
        if self._is_begin:
            return self
        self._is_begin = True
        self.emitter.begin(self._event_id, self.name, self.content.copy())
        return self

    def end(self):
        """
        End the duration span, this will automatically call when
        use context manager.

        If the span is not begin or already ended, this will not emit
        anything.

        end(), success() and fail() can only be called once, only the
        first call will take effect.

        example:

        with duration_span:
            ...

        or

        duration_span.end()

        Returns
        -------
        DurationSpan
            The same DurationSpan object for chain call.
        """
        if self._is_end:
            return

        if not self._is_begin:
            return

        self._is_end = True
        self.emitter.end(self._event_id, self.name, self.content)

    def stage(
        self, stage: str, content: Optional[dict] = None
    ) -> "DurationSpan":
        """
        Create a sub-span of the current span.

        Returns
        -------
        DurationSpan
            The sub-span of the current span.
        """
        if self._is_end or not self._is_begin:
            logger.error(f"Cannot create stage for ended event {self.name}")
            return SafeCallable()  # type: ignore
        stage_content = self.content.copy()
        if content is not None:
            stage_content.update(content)
        return DurationSpan(
            self.emitter, self.name + "#" + stage, stage_content
        )

    def extra_args(self, **kwargs):
        """
        Add extra args to the current event content, the args is
        given as kwargs.

        The args will be add into next begin() or end() event content.

        You should make sure the value is json serializable, or the
        event will not be exported.

        example:

        duration_span.extra_args(a=1, b=2)

        Returns
        -------
        DurationSpan
            The same DurationSpan object for chain call.
        """
        if self._is_end:
            logger.error(f"Cannot add extra args to ended event {self.name}")
            return self
        self.content.update(kwargs)
        return self

    def extra_dict(self, content: dict):
        """
        Add extra dict to the current event content, the dict is given
        as content.

        The dict will be add into next begin() or end() event content.

        You should make sure the value is json serializable, or the event
        will not be exported.

        example:

        duration_span.extra_dict({"a": 1, "b": 2})

        Returns
        -------
        DurationSpan
            The same DurationSpan object for chain call.
        """
        if self._is_end:
            logger.error(f"Cannot add extra dict to ended event {self.name}")
            return self
        self.content.update(content)
        return self

    def success(self, **kwargs):
        """
        Mark the duration span as success, this will automatically call
        when use context manager.

        end(), success() and fail() can only be called once, only the
        first call will take effect.

        Returns
        -------
        DurationSpan
            The same DurationSpan object for chain call.
        """
        self.content["success"] = True
        self.content.update(kwargs)
        self.end()

    def fail(self, error: str, **kwargs):
        """
        Mark the duration span as fail, this will automatically call when
        use context manager.

        end(), success() and fail() can only be called once, only the first
        call will take effect.

        Parameters
        ----------
        error : str
            The error message.
        """
        self.content["success"] = False
        self.content["error"] = error
        self.content.update(kwargs)
        self.end()

    def __enter__(self):
        return self.begin()

    def __exit__(self, exc_type, exc_value, tb):
        # if the exception is raised, record the error information, but the
        # business exception should be raised
        if exc_type is not None:
            if tb is not None:
                traceback_str = "".join(traceback.format_tb(tb))
                self.fail(str(exc_value) + "\n" + traceback_str)
            else:
                self.fail(str(exc_value))
            # ensure the exception is not swallowed
            raise
        else:
            self.success()


class Process(SafeAttribute, metaclass=SaveCallMeta):
    """
    Process is the base class for all predefined processes, offering instant,
    duration and custom_duration methods.

    Process also support safe call, which guarantees the exception happens in
    this sdk will not affect the user's program.

    User shouldn't use this class directly, instead, use the predefined
    classes.
    The inherited predefined classes can only use instant, duration and
    custom_duration methods.
    """

    def __init__(
        self, target: str, exporter: Optional[EventExporter] = None
    ) -> None:
        if exporter is None:
            exporter = get_default_exporter()
        self._emitter = EventEmitter(target, exporter)

    def instant(self, name: str, content: Optional[dict] = None):
        """
        Emit an instant event.

        Parameters
        ----------
        name : str
            The name of the event.
        content : Optional[dict], optional
            The content of the event, by default None
        """
        self._emitter.instant(name, content)

    def duration(
        self, name: str, content: Optional[dict] = None
    ) -> DurationSpan:
        """
        Emit a DurationSpan, which can be used to emit a duration begin
        and end event.

        Parameters
        ----------
        name : str
            The name of the event.
        content : Optional[dict], optional
            The content of the event, by default None
        """
        return DurationSpan(self._emitter, name, content)

    def custom_duration(
        self,
        clzz: Type[DurationSpan],
        name: str,
        content: Optional[dict] = None,
    ):
        """
        Emit a custom duration span, which can be used to define
        custom sub-spans.

        Parameters
        ----------
        clzz : Type[DurationSpan]
            The class of the duration span.
        name : str
            The name of the duration span.
        content : Optional[dict], optional
            The content of the duration span, by default None
        """
        return clzz(self._emitter, name, content)

    def error(self, msg: str):
        logger.error(msg)

    def info(self, msg: str):
        logger.info(msg)
