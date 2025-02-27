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
from collections import OrderedDict
from datetime import datetime

from dlrover.python.common.comm import AtorchEvent
from dlrover.python.common.event.train_event import (
    AtorchStepEvent,
    TrainEventState,
)
from dlrover.python.common.global_context import DefaultValues
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.training_event.event import EventTypeName


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
            self._step_events.popitem(last=False)

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

    def add_step_event(self, event: AtorchEvent):
        with self._lock:
            keys = list(self._step_events.keys())
            if event.timestamp is None or event.step is None:
                logger.error(f"invalid event: {event}")
                return
            elif len(keys) >= self._max_events:
                self._step_events.popitem(last=False)

            if event.type is EventTypeName.BEGIN:
                if len(keys) > 0:
                    last_event = self._step_events[keys[-1]]
                    if (
                        event.step != last_event.step
                        or event.timestamp < last_event.end_timestamp
                        or last_event.event_state
                        != TrainEventState.TRAIN_EVT_END
                    ):
                        logger.error(f"invalid step: {last_event}, {event}")
                        return

                step_event = AtorchStepEvent(
                    event.name, TrainEventState.TRAIN_EVT_BEGIN, event.step
                )
                step_event.begin_timestamp = event.timestamp
                self._step_events[event.timestamp] = step_event

            elif event.type is EventTypeName.END:
                if not len(keys):
                    logger.error(f"invalid step without BEGIN: {event}")
                    return
                last_key = keys[-1]
                step_event = self._step_events[last_key]
                if (
                    step_event.step + 1 != event.step
                    or step_event.begin_timestamp > event.timestamp
                    or step_event.event_state
                    != TrainEventState.TRAIN_EVT_BEGIN
                ):
                    logger.error(f"invalid step: {step_event}, {event}")
                    return

                step_event.end_timestamp = event.timestamp
                step_event.step = event.step
                step_event.event_state = TrainEventState.TRAIN_EVT_END
                step_event.localtime = int(datetime.now().timestamp())
                self._step_events[last_key] = step_event

            elif event.type is EventTypeName.INSTANT:
                step_event = AtorchStepEvent(
                    event.name, TrainEventState.TRAIN_EVT_ONCE, event.step
                )
                step_event.begin_timestamp = event.timestamp
                step_event.end_timestamp = event.timestamp
                self._step_events[event.timestamp] = step_event

    def add_ckpt_event(self, event: AtorchEvent):
        with self._lock:
            keys = list(self._step_events.keys())
            if not len(keys):
                logger.error("empty step events")
                return
            elif event.timestamp is None or event.step is None:
                logger.error(f"invalid event: {event}")
                return

            last_key = keys[-1]
            last_step = self._step_events[last_key]
            if last_step.end_timestamp is None:
                logger.error(f"invalid last step: {last_step}")
                return
            if (
                event.step != last_step.step
                or event.timestamp < last_step.end_timestamp
            ):
                logger.error(f"invalid last step: {last_step}, {event}")
                return

            if event.type is EventTypeName.BEGIN:
                last_step.start_ckpt(event.timestamp)
            elif event.type is EventTypeName.END:
                last_step.finish_ckpt(event.timestamp)
                last_step.localtime = int(datetime.now().timestamp())


class JobEventContext(Singleton):
    """
    TrainEventContext includes all events happened during training.
    """

    def __init__(self):
        self.train_steps: StepEvents = StepEvents()
        self.predict_steps: StepEvents = StepEvents()
        self.hang_threshold = DefaultValues.MAX_HANG_THRESHOLD

    def check_job_step_hang(self):
        """
        If no step events happened after hang_threshold, job may hang
        """
        now = int(datetime.now().timestamp())

        # check if step is still active
        step = self.train_steps.get_last_step_event()
        if step is None:
            return False
        else:
            if now - step.localtime < self.hang_threshold:
                return False

        # if ckpt is active, job may not be hang
        if step.ckpt_start:
            if not step.ckpt_finish:
                return False
            elif now - step.localtime < self.hang_threshold:
                return False

        # check if trainer is doing evaluation or prediction
        predict_step = self.predict_steps.get_last_step_event()
        if predict_step is None:
            logger.info(f"Detect hang on {now}: last step {step}")
            return True
        else:
            if now - predict_step.localtime < self.hang_threshold:
                return False
            else:
                logger.info(
                    f"Detect hang on {now}: last step {step}"
                    f"last predict step {predict_step.localtime}"
                )
                return True

    def check_job_step_slow(self):
        return False


def get_job_event_context() -> JobEventContext:
    event_context = JobEventContext.singleton_instance()
    return event_context
