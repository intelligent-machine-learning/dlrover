# Copyright 2023 The TFPlus Authors. All rights reserved.
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
"""constants"""


class Constants:
  NORMAL_RESTORE = 0
  MERGE_RESTORE = 1
  REPARTITION_RESTORE = 2
  REPARTITION_MERGE_RESTORE = 3

OPT_NAMES = [
    "Adagrad",
    "Adam",
    "GroupAdam",
    "GradientDescent",
    "SparseGroupFtrl",
    "RectifiedAdam",
]
