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
from typing import List

from dlrover.python.master.diagnosis.diagnosis_data import DataManager
from dlrover.python.master.diagnosis.inferencechain.common import Inference
from dlrover.python.master.diagnosis.inferencechain.inference_chain import (
    InferenceChain,
)


class Diagnostician:
    def __init__(self, data_mgr: DataManager):
        self.data_manager = data_mgr
        self.training_problems: List[Inference] = []

    def register_problems(self, problems: List[Inference]):
        self.training_problems = problems

    def observe_training(self) -> List[Inference]:
        ic = InferenceChain(self.data_manager, self.training_problems)
        return ic.infer()

    def diagnose_failure(self, inference: Inference) -> List[Inference]:
        pass
