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

from enum import Enum
from typing import Optional

from dlrover.python.training_event.emitter import DurationSpan
from dlrover.python.training_event.predefined.common import CommonPredefined


class TrainerEventName(Enum):
    TRAIN_START = "#start"
    TRAIN_INIT = "#init"
    TRAIN_LOAD_DATASET = "#load_dataset"
    TRAIN_LOAD_CKPT = "#load_ckpt"
    TRAIN_PERSIST_CKPT = "#persist_ckpt"
    TRAIN_FINISH = "#finish"

    # event from transformers trainer hooks
    TRAIN_INIT_END = "#init_end"
    TRAIN = "#train"
    EPOCH = "#epoch"
    TRAIN_STEP = "#step"
    TRAIN_SUBSTEP = "#substep"
    EVALUATE = "#evaluate"
    PREDICT = "#predict"
    PREDICT_STEP = "#predict_step"
    SAVE = "#save"
    LOG = "#log"


class TrainInitSpan(DurationSpan):
    """
    The span of the training initialization process, it has two predefined
    sub-spans:
    - load_ckpt: load the checkpoint.
    - load_dataset: load the dataset.

    You can also add custom sub-spans with stage() method.
    """

    def load_ckpt(self) -> DurationSpan:
        """
        load_ckpt event stands for the training process is loading the
        checkpoint.
        """
        return self.stage(TrainerEventName.TRAIN_LOAD_CKPT.value)

    def load_dataset(self) -> DurationSpan:
        """
        load_dataset event stands for the training process is loading
        the dataset.
        """
        return self.stage(TrainerEventName.TRAIN_LOAD_DATASET.value)


class TrainerProcess(CommonPredefined):
    """
    Trainer predefined events.
    """

    def __init__(self, target: str) -> None:
        super().__init__(target)

    # event from transformers trainer hooks

    def init_end(self, **kwargs):
        self.instant(TrainerEventName.TRAIN_INIT_END.value, kwargs)

    def train(self, **kwargs) -> DurationSpan:
        return self.duration(TrainerEventName.TRAIN.value, kwargs)

    def epoch(self, **kwargs) -> DurationSpan:
        return self.duration(TrainerEventName.EPOCH.value, kwargs)

    def step(self, global_step: int, **kwargs) -> DurationSpan:
        return self.duration(
            TrainerEventName.TRAIN_STEP.value,
            {"global_step": global_step, **kwargs},
        )

    def substep(self, global_step: int, **kwargs):
        self.instant(
            TrainerEventName.TRAIN_SUBSTEP.value,
            {"global_step": global_step, **kwargs},
        )

    def evaluate(self, global_step: int, **kwargs) -> DurationSpan:
        return self.duration(
            TrainerEventName.EVALUATE.value,
            {"global_step": global_step, **kwargs},
        )

    def predict(self, global_step: int, **kwargs) -> DurationSpan:
        return self.duration(
            TrainerEventName.PREDICT.value,
            {"global_step": global_step, **kwargs},
        )

    def predict_step(self, global_step: int, **kwargs):
        self.instant(
            TrainerEventName.PREDICT_STEP.value,
            {"global_step": global_step, **kwargs},
        )

    def save(self, global_step: int, **kwargs) -> DurationSpan:
        return self.duration(
            TrainerEventName.SAVE.value, {"global_step": global_step, **kwargs}
        )

    def log(self, global_step: int, logs: dict, **kwargs):
        self.instant(
            TrainerEventName.LOG.value,
            {"global_step": global_step, "logs": logs, **kwargs},
        )

    # event for training framework

    def start(self, params: Optional[dict] = None, **kwargs):
        """
        start event stands for the training process is started.

        Parameters
        ----------
        params : dict, optional
            The important parameters of the training process, by default None.
        """
        if params is None:
            params = {}
        self.instant(
            TrainerEventName.TRAIN_START.value, {"params": params, **kwargs}
        )

    def init(self, **kwargs) -> TrainInitSpan:
        """
        init event stands for the training initialization process.

        It has two predefined sub-spans:
        - load_ckpt: load the checkpoint.
        - load_dataset: load the dataset.

        Returns
        -------
        TrainInitSpan
            The span of the training initialization process.
        """
        return self.custom_duration(
            TrainInitSpan, TrainerEventName.TRAIN_INIT.value, kwargs
        )

    def persist_ckpt(
        self,
        global_step: int,
        path: str = "",
        size: Optional[int] = None,
        **kwargs,
    ):
        """
        persist_ckpt event used in async checkpoint saving. stands
        for the training process is persisting
        the checkpoint in an async thread.

        Parameters
        ----------
        global_step : int
            The global step of the training process.
        path : str
            The path of the checkpoint.
        size : int, optional
            The size of the checkpoint in bytes, by default None.
        """
        return self.duration(
            TrainerEventName.TRAIN_PERSIST_CKPT.value,
            {
                "global_step": global_step,
                "path": path,
                "size": size or 0,
                **kwargs,
            },
        )

    def finish(self, **kwargs):
        """
        finish event stands for the training process is successfully finished.
        """
        self.instant(TrainerEventName.TRAIN_FINISH.value, kwargs)
