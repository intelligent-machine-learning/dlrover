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
"""tfplus common"""
from __future__ import absolute_import, division, print_function

import os

from tensorflow.python.platform import tf_logging as logging

from tfplus.flash_attn.python.ops import flash_attn_ops
from tfplus.kv_variable.python import training as train
from tfplus.kv_variable.python.ops import kv_variable_ops, variable_scope
from tfplus.kv_variable.python.ops.kv_variable_ops import (
    get_kv_feature_size,
    set_tfplus_saver_mode,
    tfplus_saver_mode,
)
from tfplus.kv_variable.python.ops.variable_scope import get_kv_variable
from tfplus.version import __version__
