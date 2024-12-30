#!/bin/bash
# Copyright 2024 The DLRover Authors. All rights reserved.
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


GIT_COMMIT=$(git rev-parse HEAD)
echo "STABLE_GIT_COMMIT $GIT_COMMIT"
echo "STABLE_BUILD_TIME \"$(date '+%Y-%m-%d %H:%M:%S')\""
echo "STABLE_BUILD_TYPE $1"
echo "STABLE_BUILD_PLATFORM $(cat .build_platform)"
echo "STABLE_BUILD_PLATFORM_VERSION $(cat .platform_version)"
