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

import unittest

from dlrover.python.diagnosis.datacollector.atorch_event_collector import (
    AtorchEventCollector,
)


class TrainingEventTest(unittest.TestCase):
    def setUp(self):
        self._collector = AtorchEventCollector().singleton_instance()
        pass

    def tearDown(self):
        pass

    def test_atorch_event_collector(self):
        line = (
            "[2025-01-22T19:01:19.422454] [2044] [322] [AtorchTrainerV2] "
            '[#step] [BEGIN] {"global_step": 30, "epoch": 0.004}'
        )
        tuple = self._collector.parse_line(line)
        self.assertEqual(tuple[4], 30)
