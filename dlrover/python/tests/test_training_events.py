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
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from dlrover.python.diagnosis.datacollector.atorch_event_collector import (
    AtorchEventCollector,
    AtorchInvalidException,
    AtorchNotFoundException,
)
from dlrover.python.training_event.event import EventTargetName, EventTypeName
from dlrover.python.training_event.predefined.trainer import TrainerEventName


class TrainingEventTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_atorch_parse_line(self):
        collector = AtorchEventCollector()
        line = (
            "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
            '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004}'
        )
        events = collector.parse_line(line)
        self.assertEqual(
            events[0],
            datetime.fromisoformat("2025-01-22T19:01:19").timestamp(),
        )
        self.assertEqual(events[1], EventTargetName.TRAINER)
        self.assertEqual(events[2], TrainerEventName.TRAIN_STEP.value)
        self.assertEqual(events[3], EventTypeName.BEGIN)
        self.assertEqual(events[4], 30)

        with self.assertRaises(ValueError):
            line = (
                "[2025-01-22T1901:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004}'
            )
            collector.parse_line(line)

        with self.assertRaises(AtorchNotFoundException):
            line = "global_step 30"
            collector.parse_line(line)

        with self.assertRaises(AtorchInvalidException):
            line = "[2044] [322] [AtorchTrainerV2] global_step 30"
            collector.parse_line(line)

        with self.assertRaises(AtorchNotFoundException):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [WHOAMI] "
                '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004}'
            )
            collector.parse_line(line)

        with self.assertRaises(AtorchNotFoundException):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#ssstep] [BEGIN] {"global_step": 30, "epoch": 0.004}'
            )
            collector.parse_line(line)

        with self.assertRaises(AtorchNotFoundException):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [STOP] {"global_step": 30, "epoch": 0.004}'
            )
            collector.parse_line(line)

        with self.assertRaises(SyntaxError):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004>'
            )
            collector.parse_line(line)

        with self.assertRaises(ValueError):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [BEGIN] ("global_step", "epoch")'
            )
            collector.parse_line(line)

        with self.assertRaises(KeyError):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [BEGIN] {"step": 30, "epoch": 0.004}'
            )
            collector.parse_line(line)

    def test_atorch_collect_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = AtorchEventCollector(
                filepath=tmpdir,
                local_world_size=1,
                retry_timeout=1,
            )

            with self.assertRaises(FileNotFoundError):
                collector._monitor_file(os.path.join(tmpdir, "events_0.log"))

            filepath = os.path.join(tmpdir, "events_0.log")
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004}'
            )
            line2 = (
                "[2025-01-22T19:02:30.422454] [2044] [323] [AtorchTrainerV2] "
                '[#step] [END] {"global_step": 30, "epoch": 0.004}'
            )
            with open(filepath, "w+") as f:
                f.write(line + "\n")

            with mock.patch(
                "dlrover.python.diagnosis.datacollector."
                "atorch_event_collector.AtorchEventCollector._report_event",
                side_effect=[StopIteration("test")],
            ) as mocked_func:
                try:
                    collector._monitor_file(filepath)
                except StopIteration:
                    pass
                self.assertEqual(mocked_func.call_count, 1)

            with open(filepath, "w+") as f:
                f.write(line + "\n")
                f.write(line2 + "\n")
            with mock.patch(
                "dlrover.python.diagnosis.datacollector."
                "atorch_event_collector.AtorchEventCollector._report_event",
                side_effect=[None, StopIteration("test")],
            ) as mocked_func:
                try:
                    collector._monitor_file(filepath)
                except StopIteration:
                    pass
                self.assertEqual(mocked_func.call_count, 2)

    def test_atorch_collectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = AtorchEventCollector(
                filepath=tmpdir,
                local_world_size=1,
                retry_timeout=1,
            )

            collector.start_collectors()
            self.assertEqual(collector._threads[0].is_alive(), True)
            collector.stop_collectors()
