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

from abc import ABCMeta
from datetime import datetime

from dlrover.python.training_event.predefined.trainer import TrainerEventName


class TrainEventName(object):
    # instant event
    TRAIN_EVT_START = TrainerEventName.TRAIN_START
    TRAIN_EVT_INIT_END = TrainerEventName.TRAIN_INIT_END

    # duration event
    TRAIN_EVT_TRAIN = TrainerEventName.TRAIN
    TRAIN_EVT_EVALUATE = TrainerEventName.EVALUATE
    TRAIN_EVT_PREDICT = TrainerEventName.PREDICT
    TRAIN_EVT_STEP = TrainerEventName.TRAIN_STEP
    TRAIN_EVT_PREDICT_STEP = TrainerEventName.PREDICT_STEP
    TRAIN_EVT_FLASH_CKPT = TrainerEventName.SAVE
    TRAIN_EVT_PERSIST_CKPT = TrainerEventName.TRAIN_PERSIST_CKPT


class TrainEventState(object):
    TRAIN_EVT_ONCE = "once"
    TRAIN_EVT_BEGIN = "begin"
    TRAIN_EVT_END = "end"


class TrainEvent(metaclass=ABCMeta):
    def __init__(self, evt_name, evt_state):
        self.event_name = evt_name
        self.event_state = evt_state
        self.begin_timestamp = None
        self.end_timestamp = None
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
        self.begin_timestamp = None
        self.end_timestamp = None


class AtorchEvaluateEvent(TrainEvent):
    def __init__(
        self,
        evt_name,
        evt_state,
    ):
        super().__init__(evt_name, evt_state)
        self.event_name = evt_name
        self.event_state = evt_state
        self.begin_timestamp = None
        self.end_timestamp = None


class AtorchPredictEvent(TrainEvent):
    def __init__(
        self,
        evt_name,
        evt_state,
    ):
        super().__init__(evt_name, evt_state)
        self.event_name = evt_name
        self.event_state = evt_state
        self.begin_timestamp = None
        self.end_timestamp = None


class AtorchStepEvent(TrainEvent):
    def __init__(self, evt_name, evt_state, step):
        super().__init__(evt_name, evt_state)
        self.event_state = evt_state
        self.event_name = evt_name
        self.step = step
        self.begin_timestamp = None
        self.end_timestamp = None
        self.ckpt_start = None
        self.ckpt_finish = None

    def start_ckpt(self, timestamp: int):
        self.ckpt_start = timestamp

    def finish_ckpt(self, timestamp: int):
        if timestamp >= self.ckpt_start:
            self.ckpt_finish = timestamp
