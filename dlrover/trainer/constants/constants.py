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


class Constant(object):
    def __init__(self, name, default=None):
        self._name = name
        self._default = default

    def __call__(self):
        return self._default

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, val):
        self._default = val

    @property
    def name(self):
        return self._name


class Constants(object):
    Executor = Constant("executor")
    PsExecutor = Constant("ps_executor")
