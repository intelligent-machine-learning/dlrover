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
"""adgrad for tfplus and tensorflow"""
from __future__ import absolute_import, division, print_function

from tensorflow.python.ops import math_ops
from tensorflow.python.training.adagrad import \
    AdagradOptimizer as TFAdagradOptimizer

from tfplus.kv_variable.python.ops.kv_variable_ops import (
    KvVariable,
    gen_kv_variable_ops,
)


class AdagradOptimizer(TFAdagradOptimizer):
  """ "AdagradOptimizer inherits tf.train.AdagradOptimizer.
    We only override _resource_apply_sparse for KvVariable.
    AdagradOptimizer will both support tensorflow
    Variable and our KvVariable
    """

  def _resource_apply_sparse(self, grad, var, indices):
    if not isinstance(var, KvVariable):
      return super(AdagradOptimizer,
                   self)._resource_apply_sparse(grad, var, indices)
    acc = self.get_slot(var, "accumulator")
    return gen_kv_variable_ops.kv_variable_sparse_apply_adagrad(
        var.handle,
        acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad,
        indices,
        use_locking=True,
    )  # TODO (tongsuo): We should revolve dead-lock when use_locking set True
