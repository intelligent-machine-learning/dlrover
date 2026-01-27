# Copyright 2026 The DLRover Authors. All rights reserved.
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

from enum import Enum


class ResourceType(Enum):
    CPU = "CPU"
    GPU = "GPU"


class FailoverStrategy(Enum):
    NORMAL_FAILOVER = "NORMAL_FAILOVER"
    NODE_FAILOVER = "NODE_FAILOVER"
    GLOBAL_FAILOVER = "GLOBAL_FAILOVER"
    ABORTION_FAILOVER = "ABORTION_FAILOVER"
