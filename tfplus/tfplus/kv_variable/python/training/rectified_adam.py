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
"""Rectified Adam (RAdam) optimizer TFPlus. The file is based on
https://github.com/tensorflow/addons/blob/v0.6.0/tensorflow_addons/optimizers/rectified_adam.py
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, state_ops
from tensorflow.python.platform import tf_logging as logger
from tensorflow.python.training import adam as tf_adam


class RectifiedAdamOptimizer(tf_adam.AdamOptimizer):
  """Variant of the Adam optimizer whose adaptive learning rate is rectified
    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).
    Example of usage:
    ```python
    opt = tfplus.train.RectifiedAdamOptimizer(learning_rate=1e-3)
    ```
    Note: `amsgrad` is not described in the original paper. Use it with
          caution.
    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:
    ```python
    opt = tfplus.train.RectifiedAdamOptimizer(
        learning_rate=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
    )
    ```
    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.
    Lookahead, proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
    which is announced by Less Wright and the new combined optimizer can also
    be called "Ranger". The mechanism can be enabled by using the lookahead
    wrapper. For example:
    ```python
    radam = tfplus.train.RectifiedAdamOptimizer()
    ranger = tfplus.train.LookaheadOptimizer(radam,
              sync_period=6, slow_step_size=0.5)
    ```
    """

  def __init__(
      self,
      learning_rate=0.001,
      beta1=0.9,
      beta2=0.999,
      epsilon=1e-7,
      decay=0.0,
      weight_decay=0.0,
      amsgrad=False,
      sma_threshold=5.0,
      total_steps=0,
      warmup_proportion=0.1,
      min_lr=0.0,
      use_locking=False,
      use_nesterov=False,
      name="RectifiedAdam",
  ):
    """Construct a new RAdam optimizer.
        Args:
            learning_rate: A Tensor or a floating point value.
                The learning rate.
            beta1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            sma_threshold. A float value.
                The threshold for simple mean average.
            total_steps: An integer. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value.
                The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            use_locking: If True use locks for update operations.
            name: Optional name for the operations created when applying
                gradients. Defaults to "RectifiedAdam".
        """
    super(RectifiedAdamOptimizer, self).__init__(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        use_locking=use_locking,
        name=name,
    )
    self._initial_decay = decay
    self._weight_decay = weight_decay
    self._sma_threshold = sma_threshold
    self._warmup_proportion = warmup_proportion
    self._min_lr = min_lr
    self._amsgrad = amsgrad
    self._use_nesterov = use_nesterov
    self._initial_weight_decay = weight_decay
    self._initial_total_steps = float(total_steps)

    self._decay_t = None
    self._weight_decay_t = None
    self._sma_threshold_t = None
    self._warmup_proportion_t = None
    self._min_lr_t = None
    self._total_steps_t = None

  # pylint: disable=missing-docstring
  def _create_slots(self, var_list):
    # Create the "m", "v" slots and beta1_power, beta2_power from Adam
    super(RectifiedAdamOptimizer, self)._create_slots(var_list)
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=1.0,
                                   name="step",
                                   colocate_with=first_var)
    # Create the "accum", "linear" slots.
    if self._amsgrad:
      for v in var_list:
        with ops.colocate_with(v):
          self._zeros_slot(v, "vhat", self._name)

  # pylint: disable=missing-docstring
  def _prepare(self):
    super(RectifiedAdamOptimizer, self)._prepare()
    initial_total_steps = self._call_if_callable(self._initial_total_steps)
    warmup_proportion = self._call_if_callable(self._warmup_proportion)
    min_lr_t = self._call_if_callable(self._min_lr)
    sma_threshold = self._call_if_callable(self._sma_threshold)
    initial_weight_decay = self._call_if_callable(self._initial_weight_decay)
    initial_decay = self._call_if_callable(self._initial_decay)

    self._total_steps_t = ops.convert_to_tensor(initial_total_steps,
                                                name="total_step")
    self._warmup_proportion_t = ops.convert_to_tensor(warmup_proportion,
                                                      name="warmup_proportion")
    self._min_lr_t = ops.convert_to_tensor(min_lr_t, name="min_lr")
    self._sma_threshold_t = ops.convert_to_tensor(sma_threshold,
                                                  name="sma_threshold")
    self._weight_decay_t = ops.convert_to_tensor(initial_weight_decay,
                                                 name="weight_decay")
    self._decay_t = ops.convert_to_tensor(initial_decay, name="decay")

  def _decayed_lr(self, step, var_dtype):
    """Get decayed learning rate as a Tensor with dtype=var_dtype."""
    lr_t = math_ops.cast(self._lr_t, var_dtype)
    if self._initial_decay > 0.0:
      local_step = math_ops.cast(step, var_dtype)
      lr_t = lr_t / (1.0 + math_ops.cast(self._decay_t, var_dtype) * local_step)
    return lr_t

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (
          self._get_non_slot_variable("step", graph=graph),
          self._get_non_slot_variable("beta1_power", graph=graph),
          self._get_non_slot_variable("beta2_power", graph=graph),
      )

  def _apply_dense(self, grad, var):
    return self._tfplus_apply_dense_shared(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._tfplus_apply_dense_shared(grad, var)

  def _tfplus_apply_dense_shared(self, grad, var):
    var_dtype = var.dtype.base_dtype
    local_step, beta_1_power, beta_2_power = self._get_beta_accumulators()
    lr_t = self._decayed_lr(local_step, var_dtype)

    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta_1_t = math_ops.cast(self._beta1_t, var_dtype)
    beta_2_t = math_ops.cast(self._beta2_t, var_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var_dtype)

    if self._initial_total_steps > 0:
      total_steps = math_ops.cast(self._total_steps_t, var_dtype)
      warmup_steps = total_steps * math_ops.cast(self._warmup_proportion_t,
                                                 var_dtype)
      min_lr = math_ops.cast(self._min_lr_t, var_dtype)
      decay_steps = tf.maximum(total_steps - warmup_steps, 1)
      decay_rate = (min_lr - lr_t) / decay_steps
      lr_t = tf.where(
          local_step <= warmup_steps,
          lr_t * (local_step / warmup_steps),
          lr_t +
          decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
      )

    sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
    sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

    m_t = state_ops.assign(
        m,
        beta_1_t * m + grad * (1 - beta_1_t),
        use_locking=self._use_locking,
    )
    if self._use_nesterov:
      # refer to https://github.com/tensorflow/tensorflow/blob/ \
      # 769eddaf479c8debead9a59a72617d6ed6f0fe10/tensorflow/core/kernels/ \
      # training_ops.cc
      logger.info(f"use_nesterov is: {self._use_nesterov}")
      m_t = grad * (1.0 - beta_1_t) + beta_1_t * m_t
    m_corr_t = m_t / (1.0 - beta_1_power)

    # $$v_t = beta2 * v + (1 - beta2) * (g_t * g_t)$$
    v_t = state_ops.assign(
        v,
        beta_2_t * v + tf.square(grad) * (1.0 - beta_2_t),
        use_locking=self._use_locking,
    )

    if self._amsgrad:
      vhat = self.get_slot(var, "vhat")
      vhat_t = state_ops.assign(vhat,
                                tf.maximum(vhat, v_t),
                                use_locking=self._use_locking)
      v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
    else:
      vhat_t = None
      v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

    r_t = tf.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) /
                  (sma_inf - 2.0) * sma_inf / sma_t)

    sma_threshold = math_ops.cast(self._sma_threshold_t, var_dtype)
    var_t = tf.where(
        sma_t >= sma_threshold,
        r_t * m_corr_t / (v_corr_t + epsilon_t),
        m_corr_t,
    )
    if self._initial_weight_decay > 0.0:
      var_t += math_ops.cast(self._weight_decay_t, var_dtype) * var
    var_update = state_ops.assign_sub(var,
                                      var_t * lr_t,
                                      use_locking=self._use_locking)

    updates = [var_update, m_t, v_t]
    if self._amsgrad:
      updates.append(vhat_t)
    return tf.group(*updates)

  def _apply_sparse(self, grad, var):
    """Overriden function to apply sparse updates to Variable"""
    return self._tfplus_apply_sparse_shared(grad.values, var, grad.indices)

  def _resource_apply_sparse(self, grad, var, indices):
    """Overriden function to apply sparse updates to ResourceVariable"""
    return self._tfplus_apply_sparse_shared(grad, var, indices)

  def _tfplus_apply_sparse_shared(self, grad, var, indices):
    var_dtype = var.dtype.base_dtype
    local_step, beta_1_power, beta_2_power = self._get_beta_accumulators()
    lr_t = self._decayed_lr(local_step, var_dtype)

    beta_1_t = math_ops.cast(self._beta1_t, var_dtype)
    beta_2_t = math_ops.cast(self._beta2_t, var_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var_dtype)
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")

    if self._initial_total_steps > 0:
      total_steps = math_ops.cast(self._total_steps_t, var_dtype)
      warmup_steps = total_steps * math_ops.cast(self._warmup_proportion_t,
                                                 var_dtype)
      min_lr = math_ops.cast(self._min_lr_t, var_dtype)
      decay_steps = tf.maximum(total_steps - warmup_steps, 1)
      decay_rate = (min_lr - lr_t) / decay_steps
      lr_t = tf.where(
          local_step <= warmup_steps,
          lr_t * (local_step / warmup_steps),
          lr_t +
          decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
      )

    sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
    sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

    m_scaled = beta_1_t * array_ops.gather(m, indices)
    m_scaled_g_values = grad * (1 - beta_1_t)
    m_updates = m_scaled + m_scaled_g_values
    with ops.control_dependencies([m_updates]):
      m_t = state_ops.scatter_update(m,
                                     indices,
                                     m_updates,
                                     use_locking=self._use_locking)
    if self._use_nesterov:
      logger.info(f"use_nesterov is: {self._use_nesterov}")
      m_updates = grad * (1.0 - beta_1_t) + beta_1_t * m_updates
    m_corr_t = m_updates / (1.0 - beta_1_power)

    # $$v_t = beta2 * v + (1 - beta2) * (g_t * g_t)$$
    v_scaled = beta_2_t * array_ops.gather(v, indices)
    v_scaled_g_values = tf.square(grad) * (1 - beta_2_t)
    v_updates = v_scaled + v_scaled_g_values
    with ops.control_dependencies([v_updates]):
      v_t = state_ops.scatter_update(v,
                                     indices,
                                     v_updates,
                                     use_locking=self._use_locking)

    if self._amsgrad:
      vhat = self.get_slot(var, "vhat")
      vhat_updates = tf.maximum(v_updates, array_ops.gather(vhat, indices))
      with ops.control_dependencies([vhat_updates]):
        vhat_t = state_ops.scatter_update(vhat,
                                          indices,
                                          vhat_updates,
                                          use_locking=self._use_locking)
      v_corr_t = tf.sqrt(vhat_updates / (1.0 - beta_2_power))
    else:
      vhat_t = None
      v_corr_t = tf.sqrt(v_updates / (1.0 - beta_2_power))

    r_t = tf.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) /
                  (sma_inf - 2.0) * sma_inf / sma_t)

    sma_threshold = math_ops.cast(self._sma_threshold_t, var_dtype)
    with ops.control_dependencies([m_updates, v_updates]):
      var_updates = tf.where(
          sma_t >= sma_threshold,
          r_t * m_corr_t / (v_corr_t + epsilon_t),
          m_corr_t,
      )
      if self._initial_weight_decay > 0.0:
        var_updates += math_ops.cast(
            self._weight_decay_t, var_dtype) * array_ops.gather(var, indices)
      var_update = state_ops.scatter_sub(var,
                                         indices,
                                         var_updates * lr_t,
                                         use_locking=self._use_locking)

    updates = [var_update, m_t, v_t]
    if self._amsgrad:
      updates.append(vhat_t)
    return tf.group(*updates)

  def _finish(self, update_ops, name_scope):
    with ops.control_dependencies(update_ops):
      step, beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_step = step.assign(step + 1.0, use_locking=self._use_locking)
        update_beta1 = beta1_power.assign(beta1_power * self._beta1_t,
                                          use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(beta2_power * self._beta2_t,
                                          use_locking=self._use_locking)
      return tf.group(
          *update_ops + [update_step, update_beta1, update_beta2],
          name=name_scope,
      )
