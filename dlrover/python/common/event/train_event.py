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

from abc import ABCMeta
from datetime import datetime

from dlrover.python.training_event.predefined.trainer import TrainerEventName


class TrainEventName(object):
    # instant event
    TRAIN_EVT_START = TrainerEventName.TRAIN_START.value
    TRAIN_EVT_INIT_END = TrainerEventName.TRAIN_INIT_END.value

    # duration event
    TRAIN_EVT_TRAIN = TrainerEventName.TRAIN.value
    TRAIN_EVT_EVALUATE = TrainerEventName.EVALUATE.value
    TRAIN_EVT_PREDICT = TrainerEventName.PREDICT.value
    TRAIN_EVT_STEP = TrainerEventName.TRAIN_STEP.value
    TRAIN_EVT_PREDICT_STEP = TrainerEventName.PREDICT_STEP.value
    TRAIN_EVT_FLASH_CKPT = TrainerEventName.SAVE.value
    TRAIN_EVT_PERSIST_CKPT = TrainerEventName.TRAIN_PERSIST_CKPT.value


class TrainEventState(object):
    TRAIN_EVT_ONCE = "once"
    TRAIN_EVT_BEGIN = "start"
    TRAIN_EVT_END = "finish"


class TrainEvent(metaclass=ABCMeta):
    """
    TrainEvent is an abstract base class that defines the interface for
    events come from trainer, e.g. Atorch. Users can also add their own
    events from other trainers, e.g. Megatron, and give DLRover the
    ability to trace and act on them.

    We write an example of Atorch, by using training_event sdk
    (dlrover/python/training_event), to expose key events so that DLRover
    can trace these events, and make a healthcheck on the training procedure

    By doing this, DLRover can check if job is hang, or going slow due to
    some stragglers, and make diagnosis on these.

    """

    def __init__(self, evt_name, evt_state):
        self.event_name = evt_name
        self.event_state = evt_state
        self.begin_timestamp = 0
        self.end_timestamp = 0
        self.localtime = int(datetime.now().timestamp())

    def begin(self, timestamp: int):
        self.begin_timestamp = timestamp

    def end(self, timestamp: int):
        if timestamp >= self.begin_timestamp:
            self.end_timestamp = timestamp


class AtorchTrainEvent(TrainEvent):
    def __init__(
        self,
        evt_name,
        evt_state,
    ):
        super().__init__(evt_name, evt_state)
        self.event_name = evt_name
        self.event_state = evt_state


class AtorchEvaluateEvent(TrainEvent):
    def __init__(
        self,
        evt_name,
        evt_state,
    ):
        super().__init__(evt_name, evt_state)
        self.event_name = evt_name
        self.event_state = evt_state


class AtorchPredictEvent(TrainEvent):
    def __init__(
        self,
        evt_name,
        evt_state,
    ):
        super().__init__(evt_name, evt_state)
        self.event_name = evt_name
        self.event_state = evt_state


class AtorchStepEvent(TrainEvent):
    def __init__(self, evt_name, evt_state, step):
        super().__init__(evt_name, evt_state)
        self.event_state = evt_state
        self.event_name = evt_name
        self.step = step
        self.step_time = 0

    def __repr__(self):
        attributes = [f"{k}={v!r}" for k, v in vars(self).items()]
        return f"{self.__class__.__name__}({', '.join(attributes)})"
