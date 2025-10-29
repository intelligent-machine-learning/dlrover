# Copyright 2022 The DLRover Authors. All rights reserved.
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

from dlrover.python.master.scaler.base_scaler import Scaler, ScalePlan

# Do nothing for VolcanoScaler, use volcano to scaling.
class VolcanoScaler(Scaler):
    def __init__(self, job_name, namespace):
        super(VolcanoScaler, self).__init__(job_name)
        self._namespace = namespace

    def start(self):
        pass

    def scale(self, plan: ScalePlan):
        pass
