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

import os
import re
import time
from ast import literal_eval
from datetime import datetime

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.training_event.config import Config
from dlrover.python.training_event.event import (
    Event,
    EventTargetName,
    EventTypeName,
)
from dlrover.python.training_event.predefined.trainer import TrainerEventName


class AtorchNotFoundException(Exception):
    pass


class AtorchInvalidException(Exception):
    pass


class AtorchEventCollector(Singleton):
    """
    Atorch Event Collector implement methods of events log collection

    """

    def __init__(self, filepath=Config.file_dir):
        super().__init__()
        self._prefix_pattern = r"\s*\[(.*?)\]\s*"
        self._stop_collector = False
        self._client = MasterClient.singleton_instance()
        self._filepath = filepath

    def parse_line(self, line):
        match = re.findall(self._prefix_pattern, line)
        if not match:
            raise AtorchNotFoundException()
        if len(match) != Event.max_event_prefix:
            raise AtorchInvalidException()

        dt = datetime.fromisoformat(match[0])
        event_ts = int(dt.timestamp())

        # EventTargetName
        event_target = str(match[3])
        if event_target not in (
            EventTargetName.TRAINER,
            EventTargetName.SAVER,
        ):
            raise AtorchNotFoundException()

        # TrainerEventName
        event_name = str(match[4])
        if event_name not in (
            TrainerEventName.TRAIN_STEP.value,
            TrainerEventName.SAVE.value,
        ):
            raise AtorchNotFoundException()

        # EventTypeName
        event_type = str(match[5])
        if event_type not in [
            EventTypeName.BEGIN,
            EventTypeName.END,
        ]:
            raise AtorchNotFoundException()

        text = literal_eval(re.sub(self._prefix_pattern, "", line))
        if isinstance(text, dict):
            event_step = int(text["global_step"])
        else:
            raise ValueError

        return event_ts, event_target, event_name, event_type, event_step

    def collect_events(self, rank: int):
        filepath = os.path.join(self._filepath, f"events_{rank}.log")

        with open(filepath, "r") as f:
            logger.info(f"Atorch collector is working on {filepath}")
            f.seek(0, 0)

            while True:
                if self._stop_collector:
                    logger.info("Atorch collector stopped.")
                    break

                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue

                try:
                    ts, target, event_name, event_type, step = self.parse_line(
                        line
                    )
                except (AtorchNotFoundException, AtorchInvalidException):
                    continue
                except (ValueError, KeyError) as e:
                    logger.error(f"Parse {line} error: {e}")
                except Exception as e:
                    logger.error(f"Parse {line} unexpected error: {e}")

                self._client.report_atorch_event(
                    ts, target, event_name, event_type, step
                )
