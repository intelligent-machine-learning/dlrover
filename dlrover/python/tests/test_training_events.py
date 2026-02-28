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
import time
import unittest
from datetime import datetime
from unittest import mock
from unittest.mock import patch

from dlrover.python.common.comm import AtorchEvent
from dlrover.python.common.event.context import JobEventContext, StepEvents
from dlrover.python.common.event.train_event import (
    TrainEventName,
    TrainEventState,
)
from dlrover.python.common.global_context import Context
from dlrover.python.diagnosis.datacollector.atorch_event_collector import (
    AtorchEventCollector,
    AtorchInvalidException,
    AtorchNotFoundException,
)
from dlrover.python.training_event import DLRoverAgentEvent
from dlrover.python.training_event.event import EventTargetName, EventTypeName
from dlrover.python.training_event.predefined.trainer import TrainerEventName

_event_context = JobEventContext.singleton_instance()
_dlrover_context = Context.singleton_instance()


class AgentEventTest(unittest.TestCase):
    def setUp(self):
        self.agent_evt = DLRoverAgentEvent().singleton_instance()

    def tearDown(self):
        pass

    def test_agent_events(self):
        net_check_evt = self.agent_evt.network_check(
            round=1,
            node_rank=0,
        )
        net_check_evt.begin()
        net_check_evt.success(
            result="{0: 0, 1: 0}",
            status="SUCCEEDED",
            elapsed_time="1",
        )

        net_check_evt = self.agent_evt.network_check(
            round=2,
            node_rank=1,
        )
        net_check_evt.begin()
        net_check_evt.fail(
            result="{0: 1, 1: 1}",
            status="FAILED",
            elapsed_time="1",
        )

        rdzv_evt = self.agent_evt.rendezvous(
            rendezvous_type="test",
            node_name="node0",
            node_rank=0,
            timeout=3,
        )
        rdzv_evt.begin()
        rdzv_evt.fail(
            error="fatal error",
        )

        rdzv_evt2 = self.agent_evt.rendezvous(
            rendezvous_type="test2",
            node_name="node1",
            node_rank=1,
            timeout=3,
        )
        rdzv_evt2.begin()
        rdzv_evt2.success(
            round=3,
            rank=1,
            world_size=1,
        )

        self.agent_evt.start(
            args={
                "entrypoint": "test.sh",
                "min_nodes": 1,
                "max_nodes": 2,
                "nproc_per_node": 1,
            }
        )

        self.agent_evt.exit(
            success=True,
        )

        self.agent_evt.process_succeeded(
            node_rank=0,
            return_values="{0: 0, 1: 0}",
            state="SUCCEEDED",
        )

        self.agent_evt.process_fail(
            node_rank=1,
            return_values="{0: 1, 1: 1}",
            state="FAILED",
        )

        self.agent_evt.process_restart(
            node_rank=1,
            restart_count=1,
            remaining_restarts=0,
            return_values="{0: 1, 1: 1}",
            state="FAILED",
        )

        self.agent_evt.process_restart_membership(
            node_rank=1,
            restart_count=1,
            remaining_restarts=0,
            return_values="{0: 1, 1: 1}",
            state="FAILED",
        )


class TrainingEventTest(unittest.TestCase):
    def setUp(self):
        _event_context.__init__()

    def tearDown(self):
        _event_context.__init__()

    def test_atorch_parse_line(self):
        collector = AtorchEventCollector()
        with self.assertRaises(AtorchInvalidException):
            line = (
                "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
                '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004, "step_type": "train,inspect" }'
            )
            collector.parse_line(line)

        line = (
            "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
            '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004, "step_type": "train"}'
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

    def test_atorch_parse_log(self):
        events = StepEvents()
        ckpts = StepEvents()

        def mock_report_event(self, ts, target, event_name, event_type, step):
            evt = AtorchEvent(
                timestamp=ts,
                step=step,
                target=target,
                name=event_name,
                type=event_type,
            )
            if event_name == TrainEventName.TRAIN_EVT_STEP:
                events.add_step_event(evt)
            elif event_name == TrainEventName.TRAIN_EVT_FLASH_CKPT:
                ckpts.add_ckpt_event(evt)

        with patch.object(
            AtorchEventCollector, "_report_event", mock_report_event
        ):
            collector = AtorchEventCollector(
                filepath="dlrover/python/tests/data/training_events",
                local_world_size=1,
                retry_timeout=1,
            )
            collector.start_collectors()
            time.sleep(1)
            collector.stop_collectors()

            _, step = events.pop_step_event()
            self.assertEqual(step.step, 177020)
            self.assertEqual(step.step_time, 21)
            self.assertEqual(step.event_name, TrainEventName.TRAIN_EVT_STEP)
            self.assertEqual(step.event_state, TrainEventState.TRAIN_EVT_END)
            _, step = events.pop_step_event()
            self.assertEqual(step.step, 177021)
            self.assertEqual(step.step_time, 16)
            self.assertEqual(step.event_name, TrainEventName.TRAIN_EVT_STEP)
            self.assertEqual(step.event_state, TrainEventState.TRAIN_EVT_END)

            _, step = ckpts.pop_step_event()
            self.assertEqual(step.step, 177215)
            self.assertEqual(step.step_time, 92)
            self.assertEqual(
                step.event_name, TrainEventName.TRAIN_EVT_FLASH_CKPT
            )
            self.assertEqual(step.event_state, TrainEventState.TRAIN_EVT_END)

            _, step = ckpts.pop_step_event()
            self.assertEqual(step.step, 177413)
            self.assertEqual(step.step_time, 91)
            self.assertEqual(
                step.event_name, TrainEventName.TRAIN_EVT_FLASH_CKPT
            )
            self.assertEqual(step.event_state, TrainEventState.TRAIN_EVT_END)

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
                '[#step] [BEGIN] {"global_step": 29, "epoch": 0.004}'
            )
            line2 = (
                "[2025-01-22T19:02:30.422454] [2044] [323] [AtorchTrainerV2] "
                '[#step] [END] {"global_step": 30, "epoch": 0.004}'
            )
            line3 = (
                "[2025-01-22T19:03:30.422454] [2044] [323] [AtorchTrainerV2] "
                '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004}'
            )
            line4 = (
                "[2025-01-22T19:03:45.422454] [2044] [323] [AtorchTrainerV2] "
                '[#step] [END] {"global_step": 31, "epoch": 0.004}'
            )
            with open(filepath, "w+") as f:
                f.write(line + "\n")
                f.write(line2 + "\n")

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
                f.write(line3 + "\n")
                f.write(line4 + "\n")
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

    def test_atorch_step_event(self):
        test_events = StepEvents(max_events=3)
        self.assertEqual(test_events.size(), 0)
        self.assertEqual(test_events.get_last_step_event(), None)
        self.assertEqual(test_events.get_first_step_event(), None)

        now = int(datetime.now().timestamp())
        evt0 = AtorchEvent(
            timestamp=None,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.INSTANT,
            step=0,
        )
        test_events.add_step_event(evt0)
        self.assertEqual(test_events.size(), 0)

        evt0 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.INSTANT,
            step=None,
        )
        test_events.add_step_event(evt0)
        self.assertEqual(test_events.size(), 0)

        evt0 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=1,
        )
        test_events.add_step_event(evt0)
        self.assertEqual(test_events.size(), 0)

        evt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_step_event(evt1)
        self.assertEqual(test_events.size(), 1)

        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.begin_timestamp, now)
        self.assertEqual(last_step.end_timestamp, 0)
        self.assertEqual(last_step.step, 1)

        evt2 = AtorchEvent(
            timestamp=now + 1,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=2,
        )
        test_events.add_step_event(evt2)
        self.assertEqual(test_events.size(), 1)

        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.begin_timestamp, now)
        self.assertEqual(last_step.end_timestamp, now + 1)
        self.assertEqual(last_step.step, 2)
        self.assertEqual(last_step.step_time, 1)

        evt3 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=2,
        )
        test_events.add_step_event(evt3)
        self.assertEqual(test_events.size(), 1)

        evt3 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=2,
        )
        test_events.add_step_event(evt3)
        self.assertEqual(test_events.size(), 2)

        evt3 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=2,
        )
        test_events.add_step_event(evt3)
        self.assertEqual(test_events.size(), 2)

        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.begin_timestamp, now + 10)
        self.assertEqual(last_step.end_timestamp, 0)
        self.assertEqual(last_step.step, 2)
        self.assertEqual(last_step.step_time, 0)

        evt4 = AtorchEvent(
            timestamp=now + 20,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=3,
        )
        test_events.add_step_event(evt4)
        self.assertEqual(test_events.size(), 2)

        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.begin_timestamp, now + 10)
        self.assertEqual(last_step.end_timestamp, now + 20)
        self.assertEqual(last_step.event_state, TrainEventState.TRAIN_EVT_END)
        self.assertEqual(last_step.step_time, 10)
        self.assertEqual(last_step.step, 3)

        first_step = test_events.get_first_step_event()
        self.assertNotEqual(first_step.step, last_step.step)

    def test_atorch_ckpt_event(self):
        test_events = StepEvents(max_events=2)
        now = int(datetime.now().timestamp())

        ckpt1 = AtorchEvent(
            timestamp=None,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_ckpt_event(ckpt1)
        self.assertEqual(test_events.size(), 0)

        ckpt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=None,
        )
        test_events.add_ckpt_event(ckpt1)
        self.assertEqual(test_events.size(), 0)

        ckpt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_ckpt_event(ckpt1)
        self.assertEqual(test_events.size(), 1)
        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.begin_timestamp, now)
        self.assertEqual(last_step.end_timestamp, 0)
        self.assertEqual(
            last_step.event_state, TrainEventState.TRAIN_EVT_BEGIN
        )

        ckpt1 = AtorchEvent(
            timestamp=now + 1,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_ckpt_event(ckpt1)
        self.assertEqual(test_events.size(), 1)

        ckpt1 = AtorchEvent(
            timestamp=now + 1,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.END,
            step=2,
        )
        test_events.add_ckpt_event(ckpt1)
        self.assertEqual(test_events.size(), 1)

        ckpt2 = AtorchEvent(
            timestamp=now + 1,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.END,
            step=1,
        )
        test_events.add_ckpt_event(ckpt2)
        self.assertEqual(test_events.size(), 1)
        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.begin_timestamp, now)
        self.assertEqual(last_step.end_timestamp, now + 1)
        self.assertEqual(last_step.step_time, 1)

    def test_hang_threshold(self):
        now = int(datetime.now().timestamp())
        timeslice1 = [now, now + 10, now + 20, now + 30, now + 40]
        timeslice2 = [now + 1, now + 12, now + 23, now + 34, now + 45]

        _event_context.train_steps.clear_step_events()
        for i in range(5):
            evt1 = AtorchEvent(
                timestamp=timeslice1[i],
                target=EventTargetName.TRAINER,
                name=TrainEventName.TRAIN_EVT_STEP,
                type=EventTypeName.BEGIN,
                step=i,
            )
            _event_context.train_steps.add_step_event(evt1)
            evt2 = AtorchEvent(
                timestamp=timeslice2[i],
                target=EventTargetName.TRAINER,
                name=TrainEventName.TRAIN_EVT_STEP,
                type=EventTypeName.END,
                step=i + 1,
            )
            _event_context.train_steps.add_step_event(evt2)

        self.assertEqual(_event_context.train_steps.size(), 5)
        self.assertEqual(
            _event_context.train_steps.last_steps_avg_time(last_steps=5), 3.0
        )

    def test_job_step_hang(self):
        _dlrover_context.hang_downtime = 0.02

        _event_context.train_steps.clear_step_events()
        self.assertEqual(_event_context.check_job_step_hang(), False)

        with patch(
            "dlrover.python.common.event.context.StepEvents.last_steps_avg_time"
        ) as mock_method:
            mock_method.return_value = 20
            self.assertEqual(_event_context.check_job_step_hang(), False)
            self.assertEqual(_event_context.hang_threshold, 120)

        _dlrover_context.hang_downtime = 10
        with patch(
            "dlrover.python.common.event.context.StepEvents.last_steps_avg_time"
        ) as mock_method:
            mock_method.return_value = 500
            self.assertEqual(_event_context.check_job_step_hang(), False)
            self.assertEqual(_event_context.hang_threshold, 600)

        _dlrover_context.hang_downtime = 20
        with patch(
            "dlrover.python.common.event.context.StepEvents.last_steps_avg_time"
        ) as mock_method:
            mock_method.return_value = 10
            self.assertEqual(_event_context.check_job_step_hang(), False)
            self.assertEqual(_event_context.hang_threshold, 1200)

        now = int(datetime.now().timestamp())
        evt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=1,
        )
        _event_context.train_steps.add_step_event(evt1)

        evt2 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=2,
        )
        _event_context.train_steps.add_step_event(evt2)

        self.assertEqual(_event_context.check_job_step_hang(), False)

        evt3 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=2,
        )
        _event_context.train_steps.add_step_event(evt3)
        self.assertEqual(_event_context.check_job_step_hang(), False)
        time.sleep(2)
        self.assertEqual(_event_context.check_job_step_hang(), False)

    def test_ckpt_hang(self):
        _event_context.ckpt_threshold = 1

        _event_context.ckpt_steps.clear_step_events()
        self.assertEqual(_event_context.check_ckpt_hang(), False)

        now = int(datetime.now().timestamp())
        ckpt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=2,
        )
        _event_context.ckpt_steps.add_ckpt_event(ckpt1)
        self.assertEqual(_event_context.check_ckpt_hang(), False)
        time.sleep(1.2)
        self.assertEqual(_event_context.check_ckpt_hang(), True)

        ckpt2 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.END,
            step=2,
        )
        _event_context.train_steps.add_ckpt_event(ckpt2)
        self.assertEqual(_event_context.check_job_step_hang(), False)
        time.sleep(1.2)
        self.assertEqual(_event_context.check_job_step_hang(), False)
