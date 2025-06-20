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

from dlrover.python.unified.common.enums import (
    MasterStateBackendType,
    TrainerType,
)
from dlrover.python.unified.tests.base import BaseTest


class EnumsTest(BaseTest):
    def test_trainer_type(self):
        self.assertTrue(TrainerType["USER_DEFINED"])
        self.assertTrue(TrainerType["GENERATED"])
        self.assertTrue(TrainerType["ELASTIC_TRAINING"])

        with self.assertRaises(KeyError):
            self.assertTrue(TrainerType["TEST"])

    def test_master_state_backend(self):
        self.assertTrue(MasterStateBackendType["RAY_INTERNAL"])
        self.assertTrue(MasterStateBackendType["HDFS"])

        with self.assertRaises(KeyError):
            self.assertTrue(MasterStateBackendType["LOCAL"])
