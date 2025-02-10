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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)


@dataclass
class PreCheckResult(object):

    # The default success result is 0. The other result code(>0) should be
    # defined by different pre-check operator it's self.
    result: int = 0

    # The simple description info for the result.
    result_msg: str = ""

    # Abnormal nodes
    abnormal_nodes: List[Node] = field(default_factory=list)

    def is_success(self):
        return self.result == 0


class PreCheckOperator(ABC):
    @classmethod
    def get_retry_interval_secs(cls) -> int:
        """The retry interval seconds, can be overridden in subclasses."""
        return 5

    @classmethod
    def get_retry_times(cls) -> int:
        """
        The limited retry times, can be overridden in subclasses. For most
        pre-check, the retry value should > 1(at least once retry).

        The failed action will be executed if result still not ok after
        several retry times.
        """
        return 3

    @abstractmethod
    def check(self, *args, **kwargs) -> PreCheckResult:
        """The abstraction of the main check procedure."""
        pass

    @abstractmethod
    def recover_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        """The abstraction of the procedure actions if check failed."""
        pass

    @abstractmethod
    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        """The abstraction of the actions when operator check failed."""
        pass


class NoPreCheckOperator(PreCheckOperator):
    def check(self, *args, **kwargs):
        return PreCheckResult()

    def recover_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]

    def failed_actions(self, *args, **kwargs) -> List[DiagnosisAction]:
        return [NoAction()]
