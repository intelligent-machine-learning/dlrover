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

import threading
import time
from collections import OrderedDict
from datetime import datetime

from dlrover.python.common.comm import AtorchEvent
from dlrover.python.common.event.train_event import (
    AtorchStepEvent,
    TrainEventState,
)
from dlrover.python.common.global_context import Context, DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.training_event.event import EventTypeName

_dlrover_context = Context.singleton_instance()


class StepEvents(object):
    def __init__(self, max_events=DefaultValues.MAX_TRAIN_STEPS):
        """
        _step_events: an OrderedDict of AtorchStepEvent objects,
        append into whenever the event reported from agent

        """
        self._step_events: OrderedDict[
            int,
            AtorchStepEvent,
        ] = OrderedDict()
        self._lock = threading.Lock()
        self._max_events = max_events

    def clear_step_events(self):
        with self._lock:
            self._step_events.clear()

    def pop_step_event(self):
        with self._lock:
            return self._step_events.popitem(last=False)

    def size(self):
        with self._lock:
            return len(self._step_events)

    def get_last_step_event(self):
        with self._lock:
            keys = list(self._step_events.keys())
            if len(keys) == 0:
                return None
            key = keys[-1]
            return self._step_events[key]

    def get_first_step_event(self):
        with self._lock:
            keys = list(self._step_events.keys())
            if len(keys) == 0:
                return None
            key = keys[0]
            return self._step_events[key]

    def last_steps_avg_time(self, last_steps):
        """

        Args:
            last_steps:

        Returns:

        """
        with self._lock:
            keys = list(self._step_events.keys())
            if len(keys) < last_steps:
                return None

            total_step_time = 0
            steps = 0
            for key in reversed(keys):
                if self._step_events[key].step_time:
                    total_step_time += self._step_events[key].step_time
                    steps += 1
                    if steps >= last_steps:
                        break

            return float(total_step_time / steps)

    def add_step_event(self, event: AtorchEvent):
        with self._lock:
            keys = list(self._step_events.keys())
            if event.timestamp is None or event.step is None:
                logger.warning(f"invalid event: {event}")
                return
            elif len(keys) >= self._max_events:
                self._step_events.popitem(last=False)

            if event.type == EventTypeName.BEGIN:
                if len(keys) > 0:
                    last_event = self._step_events[keys[-1]]
                    if (
                        event.step != last_event.step
                        or last_event.event_state
                        != TrainEventState.TRAIN_EVT_END
                    ):
                        logger.warning(f"invalid step: {last_event}, {event}")
                        return

                    if event.timestamp < last_event.end_timestamp:
                        logger.warning(
                            f"invalid step time: {last_event}, {event}"
                        )
                        return

                step_event = AtorchStepEvent(
                    event.name, TrainEventState.TRAIN_EVT_BEGIN, event.step
                )
                step_event.begin_timestamp = event.timestamp
                step_event.localtime = int(time.time())
                self._step_events[event.timestamp] = step_event
                logger.debug(
                    f"Add BEGIN event with {event.timestamp}, {step_event}"
                )

            elif event.type == EventTypeName.END:
                if not len(keys):
                    logger.warning(f"invalid step without BEGIN: {event}")
                    return
                last_key = keys[-1]
                step_event = self._step_events[last_key]
                if (
                    step_event.step >= event.step
                    or step_event.event_state
                    != TrainEventState.TRAIN_EVT_BEGIN
                ):
                    logger.warning(f"invalid step: {step_event}, {event}")
                    return

                if step_event.begin_timestamp > event.timestamp:
                    logger.warning(
                        f"invalid step timestamp: {step_event}, {event}"
                    )
                    return

                step_event.end_timestamp = event.timestamp
                step_event.step_time = (
                    step_event.end_timestamp - step_event.begin_timestamp
                )
                step_event.step = event.step
                step_event.event_state = TrainEventState.TRAIN_EVT_END
                step_event.localtime = int(time.time())
                self._step_events[last_key] = step_event
                logger.debug(
                    f"Add END event with {event.timestamp}, {step_event}"
                )

    def add_ckpt_event(self, event: AtorchEvent):
        with self._lock:
            keys = list(self._step_events.keys())
            if event.timestamp is None or event.step is None:
                logger.warning(f"invalid event: {event}")
                return
            elif len(keys) >= self._max_events:
                self._step_events.popitem(last=False)

            if event.type == EventTypeName.BEGIN:
                if len(keys) > 0:
                    last_event = self._step_events[keys[-1]]
                    if last_event.event_state != TrainEventState.TRAIN_EVT_END:
                        logger.warning(
                            f"invalid ckpt step: {last_event}, {event}"
                        )
                        return
                    if event.timestamp < last_event.end_timestamp:
                        logger.warning(
                            f"invalid ckpt step time: {last_event}, {event}"
                        )
                        return

                step_event = AtorchStepEvent(
                    event.name, TrainEventState.TRAIN_EVT_BEGIN, event.step
                )
                step_event.begin_timestamp = event.timestamp
                step_event.localtime = int(time.time())
                self._step_events[event.timestamp] = step_event

            elif event.type == EventTypeName.END:
                if not len(keys):
                    logger.warning(f"invalid ckpt step without BEGIN: {event}")
                    return
                last_key = keys[-1]
                step_event = self._step_events[last_key]
                if (
                    step_event.step != event.step
                    or step_event.event_state
                    != TrainEventState.TRAIN_EVT_BEGIN
                ):
                    logger.warning(f"invalid ckpt step: {step_event}, {event}")
                    return
                if step_event.begin_timestamp > event.timestamp:
                    logger.warning(
                        f"invalid ckpt step time: {step_event}, {event}"
                    )
                    return

                step_event.end_timestamp = event.timestamp
                step_event.step_time = (
                    step_event.end_timestamp - step_event.begin_timestamp
                )
                step_event.step = event.step
                step_event.event_state = TrainEventState.TRAIN_EVT_END
                step_event.localtime = int(time.time())
                self._step_events[last_key] = step_event


class JobEventContext(Singleton):
    """
    TrainEventContext includes all events happened during training.
    """

    def __init__(self):
        self.train_steps: StepEvents = StepEvents()
        self.predict_steps: StepEvents = StepEvents()
        self.ckpt_steps: StepEvents = StepEvents()
        self.hang_threshold = DefaultValues.HANG_DOWNTIME
        self.ckpt_threshold = DefaultValues.MAX_CKPT_THRESHOLD

    def check_job_step_hang(self):
        # recalc hang threshold
        avg_step_time = self.train_steps.last_steps_avg_time(
            last_steps=DefaultValues.MAX_AVG_STEPS
        )
        if avg_step_time is None:
            self.hang_threshold = _dlrover_context.hang_downtime * 60
        else:
            self.hang_threshold = 10 * avg_step_time

            if self.hang_threshold > DefaultValues.HANG_DOWNTIME * 60:
                self.hang_threshold = DefaultValues.HANG_DOWNTIME * 60
            elif self.hang_threshold < DefaultValues.MIN_HANG_DOWNTIME * 60:
                self.hang_threshold = DefaultValues.MIN_HANG_DOWNTIME * 60

        now = int(datetime.now().timestamp())

        # check if step is still active
        step = self.train_steps.get_last_step_event()
        if step is None:
            logger.warning("Skip step hang detection since no step collected")
            return False
        elif step.event_state == TrainEventState.TRAIN_EVT_BEGIN:
            if now - step.localtime < self.hang_threshold:
                logger.debug(
                    f"No hang detected: {now} {self.hang_threshold} {step}"
                )
                return False
            else:
                logger.warning(
                    f"Detect hang: {now} {self.hang_threshold} {step}"
                )
                return True
        else:
            logger.debug(f"No new step: {now} {self.hang_threshold} {step}")
            return False

    def check_ckpt_hang(self):
        now = int(datetime.now().timestamp())

        step = self.ckpt_steps.get_last_step_event()
        if step is None:
            return False
        elif step.event_state == TrainEventState.TRAIN_EVT_BEGIN:
            if now - step.localtime < self.ckpt_threshold:
                return False
            else:
                logger.warning(
                    f"Detect ckpt hang: {now} {self.ckpt_threshold} {step}"
                )
                return True
        else:
            return False


def get_job_event_context() -> JobEventContext:
    event_context = JobEventContext.singleton_instance()
    return event_context
