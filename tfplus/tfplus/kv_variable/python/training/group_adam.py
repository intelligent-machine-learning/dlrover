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
"""Adam + Group Lasso for TensorFlow and TFPlus"""
from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import constant_op, ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import adam as tf_adam

from tfplus.kv_variable.python.ops import variable_scope
from tfplus.kv_variable.python.ops.kv_variable_ops import (
    KvVariable,
    gen_kv_variable_ops,
)


class GroupAdamOptimizer(tf_adam.AdamOptimizer):
  """Optimizer that implements the Adam algorithm with support for group lasso
    This algorithm is proposed by yueyun.yy@antfin.com.
    """

  def __init__(
      self,
      learning_rate,
      initial_accumulator_value=0.0,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-8,
      l1_regularization_strength=0.0,
      l2_regularization_strength=0.0,
      l21_regularization_strength=0.0,
      use_locking=False,
      name="GroupAdam",
      accum_name=None,
      linear_name=None,
      version=4,
  ):
    """Construct a new Group Adam optimizer.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      initial_accumulator_value: The starting value for accumulators.
        Only zero or positive values are allowed.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      l21_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "GroupAdam".
      accum_name: The suffix for the variable that keeps the gradient squared
        accumulator.  If not present, defaults to name.
      linear_name: The suffix for the variable that keeps the linear gradient
        accumulator.  If not present, defaults to name + "_1".
      version: the specific version of GroupAdam.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    super(GroupAdamOptimizer, self).__init__(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=use_locking,
        name=name,
    )

    if initial_accumulator_value < 0.0:
      raise ValueError(
          "initial_accumulator_value %f needs to be be positive or zero" %
          initial_accumulator_value)
    if l1_regularization_strength < 0.0:
      raise ValueError(
          "l1_regularization_strength %f needs to be positive or zero" %
          l1_regularization_strength)
    if l2_regularization_strength < 0.0:
      raise ValueError(
          "l2_regularization_strength %f needs to be positive or zero" %
          l2_regularization_strength)
    if l21_regularization_strength < 0.0:
      raise ValueError("l21_regularization_strength %f needs to be positive"
                       " or zero" % l21_regularization_strength)

    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength
    self._l21_regularization_strength = l21_regularization_strength

    self._l1_regularization_strength_tensor = None
    self._l2_regularization_strength_tensor = None
    self._l21_regularization_strength_tensor = None
    self._l2_shrinkage_regularization_strength_tensor = None
    self._accum_name = accum_name
    self._linear_name = linear_name
    self._version = version

  # pylint: disable=missing-docstring
  def _create_slots(self, var_list):
    # Create the "m", "v" slots and beta1_power, beta2_power from Adam
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1,
        name="beta1_power",
        colocate_with=first_var,
    )
    self._create_non_slot_variable(
        initial_value=self._beta2,
        name="beta2_power",
        colocate_with=first_var,
    )
    for v in var_list:
      if not isinstance(v, KvVariable):
        # Create slots for the first and second moments.
        self._zeros_slot(v, "m", self._name)
        self._zeros_slot(v, "v", self._name)
      else:
        with ops.colocate_with(v):
          if v.has_path() and self._version < 3:
            logging.warning(
                "All slot KvVariables with ssd storage need to be merged")
          if (self._version >= 3 or v.has_path()
              or v.kv_options is variable_scope.default_kv_option()):
            v.num_concat_opt_vars = 3
            self._zeros_slot(
                v,
                "m_v_linear",
                self._linear_name or (self._name + "_3"),
            )
          elif self._version <= 2:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            val = constant_op.constant(
                self._initial_accumulator_value,
                dtype=v.dtype,
                shape=v.get_shape(),
            )
            if self._version == 2:
              self._zeros_slot(
                  v,
                  "linear",
                  self._linear_name or (self._name + "_3"),
              )
            else:
              self._get_or_make_slot(v, val, "accum", self._accum_name
                                     or self._name)
              self._zeros_slot(v, "linear", self._linear_name or self._name)
          else:
            raise ValueError("Unknown version")

  def _prepare(self):
    super(GroupAdamOptimizer, self)._prepare()
    self._l1_regularization_strength_tensor = ops.convert_to_tensor(
        self._l1_regularization_strength, name="l1_regularization_strength")
    self._l2_regularization_strength_tensor = ops.convert_to_tensor(
        self._l2_regularization_strength, name="l2_regularization_strength")
    self._l21_regularization_strength_tensor = ops.convert_to_tensor(
        self._l21_regularization_strength,
        name="l21_regularization_strength",
    )

  def _resource_apply_sparse(self, grad, var, indices):
    if not isinstance(var, KvVariable):
      return super(GroupAdamOptimizer,
                   self)._resource_apply_sparse(grad, var, indices)
    m, v, linear, m_v_linear = None, None, None, None
    if (self._version >= 3 or var.has_path()
        or var.kv_options is variable_scope.default_kv_option()):
      m_v_linear = self.get_slot(var, "m_v_linear")
    elif self._version <= 2:
      linear = self.get_slot(var, "linear")
      m = self.get_slot(var, "m")
      v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    if self._version == 4:
      return gen_kv_variable_ops.kv_variable_group_sparse_apply_adam_v4(
          var.handle,
          m_v_linear.handle,
          grad,
          indices,
          math_ops.cast(self._lr_t, grad.dtype),
          math_ops.cast(beta1_power, grad.dtype),
          math_ops.cast(beta2_power, grad.dtype),
          math_ops.cast(self._beta1_t, grad.dtype),
          math_ops.cast(self._beta2_t, grad.dtype),
          math_ops.cast(self._epsilon_t, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l21_regularization_strength_tensor, grad.dtype),
          use_locking=False,
      )
    if (self._version == 3 or var.has_path()
        or var.kv_options is variable_scope.default_kv_option()):
      return gen_kv_variable_ops.kv_variable_group_sparse_apply_adam_v3(
          var.handle,
          m_v_linear.handle,
          grad,
          indices,
          math_ops.cast(self._lr_t, grad.dtype),
          math_ops.cast(beta1_power, grad.dtype),
          math_ops.cast(beta2_power, grad.dtype),
          math_ops.cast(self._beta1_t, grad.dtype),
          math_ops.cast(self._beta2_t, grad.dtype),
          math_ops.cast(self._epsilon_t, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l21_regularization_strength_tensor, grad.dtype),
          use_locking=False,
      )
    if self._version == 2:
      return (gen_kv_variable_ops.kv_variable_group_sparse_apply_adam_new_v2(
          var.handle,
          linear.handle,
          m.handle,
          v.handle,
          grad,
          indices,
          math_ops.cast(self._lr_t, grad.dtype),
          math_ops.cast(beta1_power, grad.dtype),
          math_ops.cast(beta2_power, grad.dtype),
          math_ops.cast(self._beta1_t, grad.dtype),
          math_ops.cast(self._beta2_t, grad.dtype),
          math_ops.cast(self._epsilon_t, grad.dtype),
          math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
          math_ops.cast(self._l21_regularization_strength_tensor, grad.dtype),
          use_locking=True,
      ))
    accum = self.get_slot(var, "accum")
    return gen_kv_variable_ops.kv_variable_group_sparse_apply_adam_v2(
        var.handle,
        accum.handle,
        linear.handle,
        m.handle,
        v.handle,
        grad,
        indices,
        math_ops.cast(self._lr_t, grad.dtype),
        math_ops.cast(beta1_power, grad.dtype),
        math_ops.cast(beta2_power, grad.dtype),
        math_ops.cast(self._beta1_t, grad.dtype),
        math_ops.cast(self._beta2_t, grad.dtype),
        math_ops.cast(self._epsilon_t, grad.dtype),
        math_ops.cast(self._l1_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l2_regularization_strength_tensor, grad.dtype),
        math_ops.cast(self._l21_regularization_strength_tensor, grad.dtype),
        use_locking=True,
    )
