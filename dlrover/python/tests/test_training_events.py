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

from dlrover.python.common.comm import AtorchEvent
from dlrover.python.common.event.context import JobEventContext, StepEvents
from dlrover.python.common.event.train_event import (
    TrainEventName,
    TrainEventState,
)
from dlrover.python.diagnosis.datacollector.atorch_event_collector import (
    AtorchEventCollector,
    AtorchInvalidException,
    AtorchNotFoundException,
)
from dlrover.python.training_event import DLRoverAgentEvent
from dlrover.python.training_event.event import EventTargetName, EventTypeName
from dlrover.python.training_event.predefined.trainer import TrainerEventName

_event_context = JobEventContext.singleton_instance()


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

        evt0 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_PREDICT_STEP,
            type=EventTypeName.INSTANT,
            step=1,
        )
        test_events.add_step_event(evt0)
        self.assertEqual(test_events.size(), 1)
        test_events.clear_step_events()
        self.assertEqual(test_events.size(), 0)
        test_events.add_step_event(evt0)
        self.assertEqual(test_events.size(), 1)
        test_events.pop_step_event()
        self.assertEqual(test_events.size(), 0)

        now = int(datetime.now().timestamp())
        evt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_step_event(evt1)
        self.assertEqual(test_events.size(), 1)

        evt2 = AtorchEvent(
            timestamp=now + 1,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=1,
        )
        test_events.add_step_event(evt2)
        self.assertEqual(test_events.size(), 1)

        last_step = test_events.get_last_step_event()
        self.assertNotEqual(last_step.begin_timestamp, None)
        self.assertEqual(last_step.end_timestamp, None)
        self.assertEqual(
            last_step.event_state, TrainEventState.TRAIN_EVT_BEGIN
        )

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
        self.assertNotEqual(last_step.begin_timestamp, None)
        self.assertNotEqual(last_step.end_timestamp, None)
        self.assertEqual(last_step.event_state, TrainEventState.TRAIN_EVT_END)
        first_step = test_events.get_first_step_event()
        self.assertEqual(first_step.step, last_step.step)

        evt3 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=1,
        )
        test_events.add_step_event(evt3)
        self.assertEqual(test_events.size(), 1)

        evt3 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.BEGIN,
            step=1,
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

        evt4 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=3,
        )
        test_events.add_step_event(evt4)
        self.assertEqual(test_events.size(), 2)

        last_step = test_events.get_last_step_event()
        self.assertNotEqual(last_step.begin_timestamp, None)
        self.assertNotEqual(last_step.end_timestamp, None)
        self.assertEqual(last_step.event_state, TrainEventState.TRAIN_EVT_END)
        first_step = test_events.get_first_step_event()
        self.assertNotEqual(first_step.step, last_step.step)

    def test_atorch_ckpt_event(self):
        test_events = StepEvents(max_events=2)
        now = int(datetime.now().timestamp())

        ckpt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_ckpt_event(ckpt1)
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

        ckpt1 = AtorchEvent(
            timestamp=None,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=None,
        )
        test_events.add_ckpt_event(ckpt1)
        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.ckpt_start, None)

        ckpt1 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=1,
        )
        test_events.add_ckpt_event(ckpt1)
        last_step = test_events.get_last_step_event()
        self.assertEqual(last_step.ckpt_start, None)

        evt2 = AtorchEvent(
            timestamp=now + 1,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_STEP,
            type=EventTypeName.END,
            step=2,
        )
        test_events.add_step_event(evt2)
        self.assertEqual(test_events.size(), 1)

        ckpt1 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=2,
        )
        test_events.add_ckpt_event(ckpt1)
        last_step = test_events.get_last_step_event()
        self.assertNotEqual(last_step.ckpt_start, None)
        self.assertEqual(last_step.ckpt_finish, None)

        ckpt2 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.END,
            step=3,
        )
        test_events.add_ckpt_event(ckpt2)
        last_step = test_events.get_last_step_event()
        self.assertNotEqual(last_step.ckpt_start, None)
        self.assertEqual(last_step.ckpt_finish, None)

        ckpt2 = AtorchEvent(
            timestamp=now + 10,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.END,
            step=2,
        )
        test_events.add_ckpt_event(ckpt2)
        last_step = test_events.get_last_step_event()
        self.assertNotEqual(last_step.ckpt_start, None)
        self.assertNotEqual(last_step.ckpt_finish, None)

    def test_job_step_hang(self):
        _event_context.hang_threshold = 1
        self.assertEqual(_event_context.check_job_step_hang(), False)

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
        time.sleep(1.2)
        self.assertEqual(_event_context.check_job_step_hang(), True)

        ckpt1 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.BEGIN,
            step=2,
        )
        _event_context.train_steps.add_ckpt_event(ckpt1)
        self.assertEqual(_event_context.check_job_step_hang(), False)
        time.sleep(1.2)

        print("ckpt")
        print(_event_context.train_steps.get_last_step_event())

        self.assertEqual(_event_context.check_job_step_hang(), False)

        ckpt2 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.SAVER,
            name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            type=EventTypeName.END,
            step=2,
        )
        _event_context.train_steps.add_ckpt_event(ckpt2)

        self.assertEqual(_event_context.check_job_step_hang(), False)
        time.sleep(1.2)
        self.assertEqual(_event_context.check_job_step_hang(), True)

        evt3 = AtorchEvent(
            timestamp=now,
            target=EventTargetName.TRAINER,
            name=TrainEventName.TRAIN_EVT_PREDICT_STEP,
            type=EventTypeName.INSTANT,
            step=1,
        )
        _event_context.predict_steps.add_step_event(evt3)
        self.assertEqual(_event_context.check_job_step_hang(), False)
        time.sleep(1.2)
        self.assertEqual(_event_context.check_job_step_hang(), True)
