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
from dataclasses import dataclass


@dataclass
class FailureDesc(object):
    """
    Description of a failure to achieve elaborate fault tolerance.

    Attributes:
        is_workload (bool): Is workload failure or master failure.
            Default is True(is workload failure).
        workload_name (str, optional): The name of the workload(same with
            vertex name and actor name) who got failure. Must be specified if
            'is_workload' is True.
        failure_time (int, optional): The timestamp of failure.
        failure_level (int, optional): The severity of the failure:
            -1: not known,
            0: no need to care,
            1: partial recoverable,
            2: unrecoverable
            Default is -1.
        reason (str, optional): The failure reason in str format.
    """

    is_workload: bool = True
    workload_name: str = ""
    failure_time: int = 0
    failure_level: int = -1
    reason: str = ""
