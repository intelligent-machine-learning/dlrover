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
"""Tests for Rectified Adam optimizer."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tfplus.kv_variable.python.training.rectified_adam import (
    RectifiedAdamOptimizer,
)


class RectifiedAdamTest(tf.test.TestCase):
  """Run tests for Rectified Adam optimizer."""

  def run_dense_sample(self, iterations, expected, optimizer):
    """Run dense update by optimizer"""
    var_0 = tf.Variable([1.0, 2.0], dtype=tf.float32)
    var_1 = tf.Variable([3.0, 4.0], dtype=tf.float32)

    grad_0 = tf.constant([0.1, 0.2], dtype=tf.float32)
    grad_1 = tf.constant([0.03, 0.04], dtype=tf.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))
    if tf.executing_eagerly():
      for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)
    else:
      update = optimizer.apply_gradients(grads_and_vars)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      for _ in range(iterations):
        self.evaluate(update)

    self.assertAllClose(var_0.read_value(), expected[0], atol=2e-4)
    self.assertAllClose(var_1.read_value(), expected[1], atol=2e-4)

  def run_sparse_sample(self, iterations, expected, optimizer):
    """Run sparse update by optimizer"""
    var_0 = tf.Variable([1.0, 2.0])
    var_1 = tf.Variable([3.0, 4.0])

    grad_0 = tf.IndexedSlices(tf.constant([0.1]), tf.constant([0]),
                              tf.constant([2]))
    grad_1 = tf.IndexedSlices(tf.constant([0.04]), tf.constant([1]),
                              tf.constant([2]))

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    if tf.executing_eagerly():
      for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)
    else:
      update = optimizer.apply_gradients(grads_and_vars)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      for _ in range(iterations):
        self.evaluate(update)

    self.assertAllClose(var_0.read_value(), expected[0], atol=2e-4)
    self.assertAllClose(var_1.read_value(), expected[1], atol=2e-4)

  def test_dense_sample(self):
    # Expected values are obtained from the official implementation
    self.run_dense_sample(
        iterations=1000,
        expected=[[0.5554, 1.5549], [2.5557, 3.5557]],
        optimizer=RectifiedAdamOptimizer(learning_rate=1e-3),
    )

  def test_sparse_sample(self):
    # Expected values are obtained from the official implementation
    # Dense results should be: [-0.1929,  0.8066], [1.8075, 2.8074]
    self.run_sparse_sample(
        iterations=2000,
        expected=[[-0.1929, 2.0], [3.0, 2.8074]],
        optimizer=RectifiedAdamOptimizer(learning_rate=1e-3),
    )

  def test_dense_sample_with_amsgrad(self):
    # Expected values are obtained from the official implementation
    # `amsgrad` has no effect because the gradient is fixed
    self.run_dense_sample(
        iterations=1000,
        expected=[[0.5554, 1.5549], [2.5557, 3.5557]],
        optimizer=RectifiedAdamOptimizer(learning_rate=1e-3, amsgrad=True),
    )

  def test_sparse_sample_with_amsgrad(self):
    # Expected values are obtained from the official implementation
    # `amsgrad` has no effect because the gradient is fixed
    self.run_sparse_sample(
        iterations=2000,
        expected=[[-0.1929, 2.0], [3.0, 2.8074]],
        optimizer=RectifiedAdamOptimizer(learning_rate=1e-3, amsgrad=True),
    )

  def test_dense_sample_with_weight_decay(self):
    # Expected values are obtained from the official implementation
    self.run_dense_sample(
        iterations=1000,
        expected=[[0.5472, 1.5368], [2.5276, 3.5176]],
        optimizer=RectifiedAdamOptimizer(learning_rate=1e-3, weight_decay=0.01),
    )

  def test_sparse_sample_with_weight_decay(self):
    # Expected values are obtained from the official implementation
    # Dense results should be: [-0.2029,  0.7768], [1.7578, 2.7380]
    self.run_sparse_sample(
        iterations=2000,
        expected=[[-0.2029, 2.0], [3.0, 2.7380]],
        optimizer=RectifiedAdamOptimizer(learning_rate=1e-3, weight_decay=0.01),
    )

  def test_dense_sample_with_warmup(self):
    self.run_dense_sample(
        iterations=1000,
        expected=[[0.8041, 1.8041], [2.8041, 3.8041]],
        optimizer=RectifiedAdamOptimizer(
            learning_rate=1e-3,
            total_steps=1000,
            warmup_proportion=0.1,
            min_lr=1e-5,
        ),
    )

  def test_sparse_sample_with_warmup(self):
    self.run_sparse_sample(
        iterations=2000,
        expected=[[0.4653, 2.0], [3.0, 3.4653]],
        optimizer=RectifiedAdamOptimizer(
            learning_rate=1e-3,
            total_steps=2000,
            warmup_proportion=0.1,
            min_lr=1e-5,
        ),
    )


if __name__ == "__main__":
  test.main()
