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

from dlrover.python.unified.tests.fixtures.misc import reset_all_singletons

from .fixtures.ray_util import (
    coverage_combine,
    disable_ray_auto_init,
    shared_ray,
    tmp_ray,
)

__fixtures__ = [
    coverage_combine,  # auto-use
    disable_ray_auto_init,  # auto-use
    reset_all_singletons,  # auto-use
    shared_ray,
    tmp_ray,
]
