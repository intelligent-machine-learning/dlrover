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
#
# This package includes code from other opensource projects.

"""
The code in this directory primarily provides examples for adapting to
different open-source frameworks. All implementations can be run directly,
but they are not recommended for production use. Additionally, due to the
large amount of code, it has not been placed in the `/examples` directory.

Regardless of the algorithm framework being targeted, the key modification
points are:
1. Inherit the trainer and worker (workload) classes from dlrover's abstract
classes.
2. Retain the logic for all algorithm implementations while removing other
control layer implementations.
3. Modify and adapt the entry-point implementation according to the API
approach for compatibility.
"""
