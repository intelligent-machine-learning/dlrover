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
import time
import unittest

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    DiagnosisActionQueue,
    EventAction,
    JobAbortionAction,
    NoAction,
    NodeAction,
    JobRestartAction,
)


class DiagnosisTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_action_basic(self):
        basic_action = DiagnosisAction()
        self.assertEqual(basic_action.action_type, DiagnosisActionType.NONE)
        self.assertEqual(
            basic_action._instance, DiagnosisConstant.LOCAL_INSTANCE
        )
        self.assertFalse(basic_action.is_needed())

        event_action = EventAction(
            "info", "job", "test", "test123", {"k1": "v1"}
        )
        self.assertEqual(event_action.action_type, DiagnosisActionType.EVENT)
        self.assertEqual(
            event_action._instance, DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(event_action.event_type, "info")
        self.assertEqual(event_action.event_instance, "job")
        self.assertEqual(event_action.event_action, "test")
        self.assertEqual(event_action.event_msg, "test123")
        self.assertEqual(event_action.event_labels, {"k1": "v1"})
        self.assertTrue(event_action.is_needed())

        event_action_json = event_action.to_json()
        self.assertIsNotNone(event_action_json)

        event_action_obj = EventAction.from_json(event_action_json)
        self.assertIsNotNone(event_action_obj)
        self.assertEqual(
            event_action.event_action, event_action_obj.event_action
        )

        node_relaunch_action = NodeAction(
            node_id=1,
            node_type=NodeType.WORKER,
            node_status=NodeStatus.FAILED,
            reason="hang",
            action_type=DiagnosisActionType.MASTER_RELAUNCH_WORKER,
        )
        self.assertEqual(
            node_relaunch_action.action_type,
            DiagnosisActionType.MASTER_RELAUNCH_WORKER,
        )
        self.assertEqual(
            node_relaunch_action._instance, DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(node_relaunch_action.node_id, 1)
        self.assertEqual(node_relaunch_action.node_type, "worker")

        self.assertEqual(node_relaunch_action.node_status, NodeStatus.FAILED)
        self.assertEqual(node_relaunch_action.reason, "hang")
        self.assertTrue(event_action.is_needed())

        node_relaunch_action = NodeAction(
            node_id=1,
            node_type=NodeType.WORKER,
            node_status=NodeStatus.FAILED,
            reason="hang",
            action_type=DiagnosisActionType.RESTART_WORKER,
        )
        self.assertEqual(
            node_relaunch_action.action_type,
            DiagnosisActionType.RESTART_WORKER,
        )
        self.assertTrue(event_action.is_needed())

        job_abortion_action = JobAbortionAction("test123", "test321")
        self.assertEqual(
            job_abortion_action.action_type,
            DiagnosisActionType.JOB_ABORT,
        )
        self.assertEqual(
            job_abortion_action._instance, DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(job_abortion_action.reason, "test123")
        self.assertEqual(job_abortion_action.msg, "test321")

        job_restart_action = JobRestartAction("test123", "test321")
        self.assertEqual(
            job_restart_action.action_type,
            DiagnosisActionType.JOB_RESTART,
        )
        self.assertEqual(
            job_restart_action._instance, DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(job_abortion_action.reason, "test123")
        self.assertEqual(job_restart_action.msg, "test321")
        self.assertTrue(
            JobRestartAction.from_json(job_restart_action.to_json())
        )

    def test_action_queue(self):
        action_queue = DiagnosisActionQueue()
        action0 = EventAction("test0", expired_time_period=100000)
        action1 = EventAction("test1", expired_time_period=1)
        action2 = EventAction("test2", expired_time_period=100000)

        action_queue.add_action(action0)
        action_queue.add_action(action1)
        action_queue.add_action(action2)

        time.sleep(0.1)
        self.assertEqual(
            action_queue.next_action(instance=1).action_type,
            DiagnosisActionType.NONE,
        )
        self.assertEqual(
            action_queue.next_action(
                instance=DiagnosisConstant.MASTER_INSTANCE
            ).action_type,
            DiagnosisActionType.EVENT,
        )
        self.assertEqual(
            action_queue.next_action(
                instance=DiagnosisConstant.LOCAL_INSTANCE
            ).action_type,
            DiagnosisActionType.NONE,
        )
        self.assertEqual(
            action_queue.next_action(instance=1).action_type,
            DiagnosisActionType.NONE,
        )

        ##################################################
        action_queue.clear()

        action0 = EventAction(
            event_type="type",
            event_instance="worker",
            event_action="action",
            event_msg="msg0",
        )
        action_queue.add_action(action0)
        action1 = EventAction(
            event_type="type",
            event_instance="worker",
            event_action="action",
            event_msg="msg1",
            executable_time_period=5,
        )
        action_queue.add_action(action1)
        action2 = EventAction(
            event_type="type",
            event_instance="worker",
            event_action="action",
            event_msg="msg2",
        )
        action_queue.add_action(action2)
        action3 = EventAction(
            event_type="type",
            event_instance="worker",
            event_action="action",
            event_msg="msg2",
        )
        action_queue.add_action(action3)

        self.assertEqual(
            len(action_queue._actions[DiagnosisConstant.MASTER_INSTANCE]), 3
        )
        action = action_queue.next_action(
            instance=DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(action.event_msg, "msg0")
        action = action_queue.next_action(
            instance=DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(action.event_msg, "msg2")
        action = action_queue.next_action(
            instance=DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertTrue(isinstance(action, NoAction))
        time.sleep(5)
        action = action_queue.next_action(
            instance=DiagnosisConstant.MASTER_INSTANCE
        )
        self.assertEqual(action.event_msg, "msg1")


if __name__ == "__main__":
    unittest.main()
