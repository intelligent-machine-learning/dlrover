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
from typing import Dict

from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.enums import (
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
        job_max_restart: int,
        master_max_restart: int,
        trainer_max_restart: int,
        workload_max_restart: Dict[str, int],
    ):
        """
        Configuration(non-business part) of the job.

        Args:
            job_name (str): Name of the job.
            master_state_backend_type: The type of the master state backend.
            master_state_backend_config: The configuration of the master state
                backend, like: path and so on.
            scheduling_strategy_type: The type of scheduling strategy to
                create workloads.
            job_max_restart (int, optional): The maximum limit on the number
                of job-level restarts. Default is 10.
            master_max_restart (int, optional): The maximum limit on the
                number of master restarts. Default is 10.
            trainer_max_restart (int, optional): The maximum limit on the
                number of trainer restarts. Default is 10.
            workload_max_restart (Dict[str, int], optional): The
                maximum limit on the number of workload actor restarts.
                Default is 30.
        """

        self._job_name = job_name
        self._master_state_backend_type = master_state_backend_type
        self._master_state_backend_config = master_state_backend_config
        self._scheduling_strategy_type = scheduling_strategy_type
        self._job_max_restart = job_max_restart
        self._master_max_restart = master_max_restart
        self._trainer_max_restart = trainer_max_restart
        self._workload_max_restart = workload_max_restart

    def __repr__(self):
        return (
            "JobConfig("
            f"job_name={self._job_name}, "
            f"master_state_backend_type={self._master_state_backend_type}, "
            "master_state_backend_config="
            f"{self._master_state_backend_config}, "
            f"scheduling_strategy_type={self._scheduling_strategy_type}, "
            f"job_max_restart={self._job_max_restart}, "
            f"master_max_restart={self._master_max_restart}, "
            f"trainer_max_restart={self._trainer_max_restart}, "
            f"workload_max_restart={self._workload_max_restart})"
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

    @property
    def job_max_restart(self):
        return self._job_max_restart

    @property
    def master_max_restart(self):
        return self._master_max_restart

    @property
    def trainer_max_restart(self):
        return self._trainer_max_restart

    @property
    def workload_max_restart(self):
        return self._workload_max_restart

    def get_workload_max_restart(self, role: str):
        role = role.upper()

        if role in self._workload_max_restart:
            return self._workload_max_restart[role]
        return max(DLMasterConstant.WORKLOAD_MAX_RESTART, self.job_max_restart)

    @classmethod
    def build_from_args(cls, args):
        return JobConfig(
            args.job_name,
            args.master_state_backend_type,
            args.master_state_backend_config,
            args.scheduling_strategy_type,
            args.job_max_restart,
            args.master_max_restart,
            args.trainer_max_restart,
            args.workload_max_restart,
        )
