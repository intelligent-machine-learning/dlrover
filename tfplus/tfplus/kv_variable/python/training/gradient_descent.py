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
"""
Gradient decent algorithm for tensorflow and tfplus
"""
from __future__ import absolute_import, division, print_function

from tensorflow.python.training import gradient_descent as tf_sgd

from tfplus.kv_variable.python.ops.kv_variable_ops import scatter_add


# pylint: disable=abstract-method
class GradientDescentOptimizer(tf_sgd.GradientDescentOptimizer):
  """
    Optimizer that implements the gradient descent algorithm.
    Both support tensorflow  and tfplus.
    """

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return scatter_add(handle, indices, -grad * self._learning_rate)
