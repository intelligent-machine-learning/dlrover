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
"""Adam optimizer for TFPlus. We inherit TensorFlow AdamOptimizer to
implement a variant of AdamOptimizer, which can handle sparse updates
more efficiently than the original one. This variant is written based
on the idea proposed in LazyAdamOptimizer. Please refer to
https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/opt/python/training/lazy_adam_optimizer.py.
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import (
    array_ops,
    control_flow_ops,
    math_ops,
    state_ops,
)
from tensorflow.python.training import adam as tf_adam

from tfplus.kv_variable.python.ops import variable_scope
from tfplus.kv_variable.python.ops.kv_variable_ops import KvVariable


class AdamOptimizer(tf_adam.AdamOptimizer):
  """TFPlus Adam optimizer for efficient sparse updates"""

  def __init__(
      self,
      learning_rate=0.001,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-8,
      use_locking=False,
      name="Adam",
      version=1,
  ):
    """Construct TFPlus Adam optimizer"""
    self._version = version
    super(AdamOptimizer, self).__init__(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=use_locking,
        name=name,
    )

  def _create_slots(self, var_list):
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

    # Create slots for the first and second moments.
    for v in var_list:
      if not isinstance(v, KvVariable):
        self._zeros_slot(v, "m", self._name)
        self._zeros_slot(v, "v", self._name)
      else:
        if (self._version == 2 or v.has_path()
            or v.kv_options is variable_scope.default_kv_option()):
          v.num_concat_opt_vars = 2
          self._zeros_slot(v, "m_v", self._name)
        elif self._version <= 1:
          self._zeros_slot(v, "m", self._name)
          self._zeros_slot(v, "v", self._name)
        else:
          raise ValueError("Unknown version")

  def _tfplus_apply_sparse_shared(self, grad, var, indices):
    """Shared internal function to apply sparse updates"""
    # $$m_t = beta1 * m + (1 - beta1) * g_t$$
    if (isinstance(var, KvVariable) and var.num_concat_opt_vars == 2):  # pylint: disable=simplifiable-if-statement
      enabled_concat_opt_vars = True
    else:
      enabled_concat_opt_vars = False
    if enabled_concat_opt_vars:
      m_v = self.get_slot(var, "m_v")
      m_v_values = array_ops.gather(m_v, indices)
      embedding_dim = var.shape.as_list()[1]
      indices_shape = tf.shape(indices)
      indices_length = indices_shape[0]
      m_values = array_ops.slice(m_v_values, [0, 0],
                                 [indices_length, embedding_dim])
      v_values = array_ops.slice(m_v_values, [0, embedding_dim],
                                 [indices_length, embedding_dim])
    else:
      m = self.get_slot(var, "m")
      m_values = array_ops.gather(m, indices)
      v = self.get_slot(var, "v")
      v_values = array_ops.gather(v, indices)

    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    m_scaled = beta1_t * m_values
    m_scaled_g_values = grad * (1 - beta1_t)
    m_updates = m_scaled + m_scaled_g_values

    # $$v_t = beta2 * v + (1 - beta2) * (g_t * g_t)$$
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    v_scaled = beta2_t * v_values
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_updates = v_scaled + v_scaled_g_values

    # update m and v
    if enabled_concat_opt_vars:
      with ops.control_dependencies([m_updates, v_updates]):
        m_v_updates = array_ops.concat([m_updates, v_updates], 1)
        m_v_t = state_ops.scatter_update(m_v,
                                         indices,
                                         m_v_updates,
                                         use_locking=self._use_locking)
    else:
      with ops.control_dependencies([m_updates]):
        m_t = state_ops.scatter_update(m,
                                       indices,
                                       m_updates,
                                       use_locking=self._use_locking)
      with ops.control_dependencies([v_updates]):
        v_t = state_ops.scatter_update(v,
                                       indices,
                                       v_updates,
                                       use_locking=self._use_locking)

    # $$variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))$$
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr = lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power)
    with ops.control_dependencies([m_updates, v_updates]):
      var_updates = (lr * m_updates / (epsilon_t + math_ops.sqrt(v_updates)))
      var_update_op = state_ops.scatter_sub(var,
                                            indices,
                                            var_updates,
                                            use_locking=self._use_locking)
    if enabled_concat_opt_vars:
      return control_flow_ops.group(var_update_op, m_v_t)
    return control_flow_ops.group(var_update_op, m_t, v_t)

  def _apply_sparse(self, grad, var):
    """Overriden function to apply sparse updates to Variable"""
    return self._tfplus_apply_sparse_shared(grad.values, var, grad.indices)

  def _resource_apply_sparse(self, grad, var, indices):
    """Overriden function to apply sparse updates to ResourceVariable"""
    return self._tfplus_apply_sparse_shared(grad, var, indices)
