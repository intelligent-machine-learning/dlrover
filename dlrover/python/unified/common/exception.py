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


class InvalidJobConfiguration(Exception):
    def __init__(self):
        super(InvalidJobConfiguration, self).__init__(
            "The job configuration is invalid."
        )


class InvalidDLConfiguration(Exception):
    def __init__(self):
        super(InvalidDLConfiguration, self).__init__(
            "The deep learning configuration is invalid."
        )


class ResourceError(Exception):
    def __init__(self):
        super(ResourceError, self).__init__("The resource is not satisfied.")
