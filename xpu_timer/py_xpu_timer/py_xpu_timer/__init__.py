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

# version is genreated by bazel
import io
import sys

from .version import *

if isinstance(sys.stdout, io.TextIOWrapper) and sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8")  # type ignore[attr-defined]
print(f"git commit is {__version__}")  # type: ignore[name-defined]
print(f"build time is {__build_time__}")  # type: ignore[name-defined]
print(f"build type is {__build_type__}")  # type: ignore[name-defined]
print(f"build platform is {__build_platform__}")  # type: ignore[name-defined]
print(f"build platform version is {__build_platform_version__}")  # type: ignore[name-defined]
