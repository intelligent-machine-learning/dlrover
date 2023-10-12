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
"""Ftrl-proximal and sparse group lasso for tfplus and tensorflow"""
from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.ftrl import FtrlOptimizer as TFFtrlOptimizer

from tfplus.kv_variable.python.ops.kv_variable_ops import (
    KvVariable,
    gen_kv_variable_ops,
)


class SparseGroupFtrlOptimizer(TFFtrlOptimizer):
  """ "SparseGroupFtrlOptimizer inherits tf.compat.v1.train.FtrlOptimizer
    which implements ftrl + group lasso + sparse group lasso for
    KvVariable. If var is tensorflow variable, the optimizer will
    equal to tf.compat.v1.train.FtrlOptimizer.
    We add l21_regularization_strength parameter for L21 norm.
    Here we only override _resource_apply_sparse for KvVariable.
    SparseGroupFtrl is .
    """

  def __init__(
      self,
      learning_rate,
      learning_rate_power=-0.5,
      initial_accumulator_value=0.1,
      l1_regularization_strength=0.0,
      l2_regularization_strength=0.0,
      l21_regularization_strength=0.0,
      use_locking=False,
      name="SparseGroupFtrl",
      accum_name=None,
      linear_name=None,
      l2_shrinkage_regularization_strength=0.0,
  ):
    super(SparseGroupFtrlOptimizer, self).__init__(
        learning_rate,
        learning_rate_power=learning_rate_power,
        initial_accumulator_value=initial_accumulator_value,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        use_locking=use_locking,
        name=name,
        accum_name=accum_name,
        linear_name=linear_name,
        l2_shrinkage_regularization_strength=
        l2_shrinkage_regularization_strength,
    )
    if l21_regularization_strength < 0.0:
      raise ValueError("l21_regularization_strength %f needs to be positive"
                       " or zero" % l21_regularization_strength)
    self._l21_regularization_strength = l21_regularization_strength

  def _prepare(self):
    super(SparseGroupFtrlOptimizer, self)._prepare()
    self._l21_regularization_strength_tensor = ops.convert_to_tensor(
        self._l21_regularization_strength,
        name="l21_regularization_strength",
    )

  def _resource_apply_sparse(self, grad, var, indices):
    if not isinstance(var, KvVariable):
      return super(SparseGroupFtrlOptimizer,
                   self)._resource_apply_sparse(grad, var, indices)
    accum = self.get_slot(var, "accum")
    linear = self.get_slot(var, "linear")
    return gen_kv_variable_ops.kv_variable_sparse_group_sparse_apply_ftrl_v2(
        var.handle,
        accum.handle,
        linear.handle,
        grad,
        indices,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._adjusted_l2_regularization_strength_tensor,
                      grad.dtype),
        math_ops.cast(self._l21_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l2_shrinkage_regularization_strength_tensor,
                      grad.dtype),
        math_ops.cast(self._learning_rate_power_tensor, grad.dtype),
        use_locking=True,
    )  # TODO (tongsuo): We should revolve dead-lock when use_locking set True
