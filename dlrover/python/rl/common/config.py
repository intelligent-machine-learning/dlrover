# Copyright 2025 The EasyDL Authors. All rights reserved.
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
from typing import Dict

from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.rl.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)


class JobConfig(PickleSerializable):
    def __init__(
        self,
        job_name: str,
        master_state_backend_type: MasterStateBackendType,
        master_state_backend_config: Dict,
        scheduling_strategy_type: SchedulingStrategyType,
    ):
        """
        Configuration(non-business part) of the job.

        Args:
            job_name: Name of the job.
            master_state_backend_type: The type of the master state backend.
            master_state_backend_config: The configuration of the master state
                backend, like: path and so on.
            scheduling_strategy_type: The type of scheduling strategy to
                create workloads.
        """

        self._job_name = job_name
        self._master_state_backend_type = master_state_backend_type
        self._master_state_backend_config = master_state_backend_config
        self._scheduling_strategy_type = scheduling_strategy_type

    def __repr__(self):
        return (
            "JobConfig("
            f"job_name={self._job_name}, "
            f"master_state_backend_type={self._master_state_backend_type}, "
            "master_state_backend_config="
            f"{self._master_state_backend_config}, "
            f"scheduling_strategy_type={self._scheduling_strategy_type})"
        )

    @property
    def job_name(self) -> str:
        return self._job_name

    @property
    def master_state_backend_type(self) -> MasterStateBackendType:
        return self._master_state_backend_type

    @property
    def master_state_backend_config(self) -> Dict:
        return self._master_state_backend_config

    @property
    def scheduling_strategy_type(self) -> SchedulingStrategyType:
        return self._scheduling_strategy_type

    @classmethod
    def build_from_args(cls, args):
        return JobConfig(
            args.job_name,
            args.master_state_backend_type,
            args.master_state_backend_config,
            args.scheduling_strategy_type,
        )
