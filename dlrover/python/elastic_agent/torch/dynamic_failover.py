#  Copyright 2026 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod

from dlrover.python.common.comm import ProcessFailureInfo
from dlrover.python.common.enums import FailoverStrategy


class DynamicFailoverExtension(ABC):
    """
    Dynamic extension for fault-tolerance execution.
    """

    @abstractmethod
    def get_user_failover_strategy(
        self, failure_info: ProcessFailureInfo
    ) -> FailoverStrategy:
        """
        The user-side implementation to specify a failover-strategy to DLRover
        according to the failure info of a process. Defaults to returning
        FailoverStrategy.NORMAL_FAILOVER, which employs DLRover's internal logic.

        This implementation can be based on simple rule definitions using error
        codes or complex logic calls involving external services or model inference.

        Args:
            failure_info (ProcessFailureInfo): The basic information of process
                when failure happens.

        Returns:
            FailoverStrategy: The failover strategy.
        """

        return FailoverStrategy.NORMAL_FAILOVER
