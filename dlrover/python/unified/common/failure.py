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
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class FailureDesc(object):
    """
    Description of a failure to achieve elaborate fault tolerance.

    Attributes:
        failure_obj (str): Is workload failure or master failure or trainer
            failure. Supported: MASTER, TRAINER, WORKLOAD. Default is
            'WORKLOAD'.
        workload_name (str, optional): The name of the workload(same with
            vertex name and actor name) who got failure. Must be specified if
            'failure_obj' is 'WORKLOAD'.
        workload_role (str, optional): The type of the workload.
        failure_time (int, optional): The timestamp of failure.
        failure_level (int, optional): The severity of the failure:
            -1: not known,
            0: no need to care,
            1: partial recoverable,
            2: unrecoverable
            3: recoverable
            Default is -1.
        reason (str, optional): The failure reason in str format.
    """

    failure_obj: str = "WORKLOAD"
    workload_name: str = ""
    workload_role: str = ""
    failure_time: int = 0
    failure_level: int = -1
    reason: str = ""
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def is_master_failure(self) -> bool:
        return self.failure_obj == "MASTER"

    def is_trainer_failure(self) -> bool:
        return self.failure_obj == "TRAINER"

    def is_workload_failure(self) -> bool:
        return self.failure_obj == "WORKLOAD"
