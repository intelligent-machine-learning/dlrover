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
"""Tests for kvVariable training ops."""
from __future__ import absolute_import, division, print_function

import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.profiler import model_analyzer, option_builder
from tensorflow.python.training import training_ops

from tfplus.kv_variable.kernels.hybrid_embedding.storage_config_pb2 import *
from tfplus.kv_variable.python.ops.kv_variable_ops import \
    gen_kv_variable_ops as gen_kv_var_ops
from tfplus.kv_variable.python.ops.kv_variable_options import (
    KvOptions,
    KvStorageConfig,
)
from tfplus.kv_variable.python.ops.variable_scope import get_kv_variable
from tfplus.kv_variable.python.training.adagrad import AdagradOptimizer
from tfplus.kv_variable.python.training.adam import AdamOptimizer
from tfplus.kv_variable.python.training.group_adam import GroupAdamOptimizer
from tfplus.kv_variable.python.training.sparse_group_ftrl import (
    SparseGroupFtrlOptimizer,
)

kv_variable_v2 = gen_kv_var_ops.kv_variable_v2
init_kv_variable_v2 = gen_kv_var_ops.init_kv_variable_v2
kv_variable_gather_v2 = gen_kv_var_ops.kv_variable_gather_v2
kv_variable_gather_or_insert_v2 = (
    gen_kv_var_ops.kv_variable_gather_or_insert_v2)

# Operations to test
kv_variable_sparse_apply_ftrl = gen_kv_var_ops.kv_variable_sparse_apply_ftrl_v2


class KvVariableTrainingOpsTest(tf.test.TestCase):
  """This class implements a set of kvVariable training ops test functions"""

  def __init__(self, method_name="run_kv_variable_training_ops_test"):
    super(KvVariableTrainingOpsTest, self).__init__(methodName=method_name)

    self.container = ""
    self.shared_name = ""
    self.use_node_name_sharing = False
    self.enter_threshold = 0
    self.embedding_dim = 64
    self.key_dtype = tf.int64
    self.value_dtype = tf.float32

    # Number of rows of the initializing table
    self.init_table_rows = 1024

  def test_kv_variable_sparse_apply_ftrl(self):
    """test kv variable sparse apply ftrl"""
    # constant values
    lr = 0.01
    l1_regularization_strength = 0.0
    l2_regularization_strength = 0.0
    l2_shrinkage_regularization_strength = 0.0
    learning_rate_power = -0.5
    var_constant_value = 0.03

    # random grad
    grad_slice_size = 300
    grad_shap = (grad_slice_size, self.embedding_dim)
    grad = tf.constant(
        np.random.normal(size=(grad_slice_size, self.embedding_dim)),
        dtype=self.value_dtype,
    )
    # init range indices
    indices = tf.range(start=0, limit=grad_slice_size, dtype=self.key_dtype)

    # tf ftrl optimizer
    tf_var = tf.Variable(
        initial_value=tf.constant(var_constant_value,
                                  shape=grad_shap,
                                  dtype=self.value_dtype),
        name="test_resource_variable",
        dtype=self.value_dtype,
    )
    tf_accum = tf.Variable(
        initial_value=tf.constant(0.1, shape=grad_shap, dtype=self.value_dtype),
        name="test_resource_variable_accum",
        dtype=self.value_dtype,
    )
    tf_linear = tf.Variable(
        initial_value=tf.constant(0.0, shape=grad_shap, dtype=self.value_dtype),
        name="test_resource_variable_linear",
        dtype=self.value_dtype,
    )
    tf_ftrl_op = training_ops.resource_sparse_apply_ftrl_v2(
        tf_var.handle,
        tf_accum.handle,
        tf_linear.handle,
        grad,
        indices,
        math_ops.cast(lr, grad.dtype),
        math_ops.cast(l1_regularization_strength, grad.dtype),
        math_ops.cast(l2_regularization_strength, grad.dtype),
        math_ops.cast(l2_shrinkage_regularization_strength, grad.dtype),
        math_ops.cast(learning_rate_power, grad.dtype),
        use_locking=False,
    )

    tf_init_op = tf.compat.v1.global_variables_initializer()

    # kvVariable ftrl optimizer
    kv_var_handle = kv_variable_v2(
        container=self.container,
        shared_name=self.shared_name,
        use_node_name_sharing=self.use_node_name_sharing,
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
        enter_threshold=self.enter_threshold,
    )
    kv_var_init_table = tf.constant(
        var_constant_value,
        dtype=self.value_dtype,
        shape=[self.init_table_rows, self.embedding_dim],
    )
    kv_var_init_op = init_kv_variable_v2(kv_var_handle, kv_var_init_table)
    kv_var_accum_handle = kv_variable_v2(
        container=self.container,
        shared_name=self.shared_name,
        use_node_name_sharing=self.use_node_name_sharing,
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
        enter_threshold=self.enter_threshold,
    )
    accum_init_table = tf.constant(
        0.1,
        dtype=self.value_dtype,
        shape=[self.init_table_rows, self.embedding_dim],
    )
    init_accum_op = init_kv_variable_v2(kv_var_accum_handle, accum_init_table)
    kv_var_linear_handle = kv_variable_v2(
        container=self.container,
        shared_name=self.shared_name,
        use_node_name_sharing=self.use_node_name_sharing,
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
        enter_threshold=self.enter_threshold,
    )
    linear_init_table = tf.constant(
        0.0,
        dtype=self.value_dtype,
        shape=[self.init_table_rows, self.embedding_dim],
    )
    init_linear_op = init_kv_variable_v2(kv_var_linear_handle,
                                         linear_init_table)

    kv_var_ftrl_op = kv_variable_sparse_apply_ftrl(
        kv_var_handle,
        kv_var_accum_handle,
        kv_var_linear_handle,
        grad,
        indices,
        math_ops.cast(lr, grad.dtype),
        math_ops.cast(l1_regularization_strength, grad.dtype),
        math_ops.cast(l2_regularization_strength, grad.dtype),
        math_ops.cast(l2_shrinkage_regularization_strength, grad.dtype),
        math_ops.cast(learning_rate_power, grad.dtype),
        use_locking=False,
    )

    kv_var_gather = kv_variable_gather_or_insert_v2(kv_var_handle,
                                                    indices=indices,
                                                    dtype=self.value_dtype)

    with self.session() as sess:
      # tf ftrl op
      sess.run(tf_init_op)
      sess.run(tf_ftrl_op)

      # kvVariable ftrl op
      sess.run(kv_var_init_op)
      sess.run(init_accum_op)
      sess.run(init_linear_op)
      sess.run(kv_var_ftrl_op)

      # test variable value is equal or not
      tf_var_value = sess.run(tf_var)
      kv_var_value = sess.run(kv_var_gather)
      result = True
      if not np.allclose(tf_var_value, kv_var_value, atol=1e-8):
        result = False
      self.assertEqual(result, True)

  def init_test_data(self, h=10, w=None):
    """init variable and sparse tensor"""
    w = w or self.embedding_dim
    full_index = list(range(h))
    grad_value = np.random.rand(h, w).astype(np.float32)
    # grad_value = np.array([[0.1] * w] * h, dtype=np.float32)
    # print('grad: {}'.format(grad_value))
    kv_options = KvOptions(
        combination=StorageCombination.MEM,
        configs={
            StorageType.MEM_STORAGE: KvStorageConfig(),
        },
    )
    with tf.device("/cpu:0"):
      tf_var = tf.Variable(
          initial_value=tf.ones(shape=[h, w], dtype=tf.float32),
          name="dense_table",
      )
      resource_var = tf.Variable(
          initial_value=tf.ones(shape=[h, w], dtype=tf.float32),
          name="resource_table",
      )
      kv_var = get_kv_variable(
          "kv_table",
          embedding_dim=w,
          initializer=tf.compat.v1.ones_initializer,
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          kv_options=kv_options,
      )
      y = tf.constant(grad_value)
      full_indices = tf.constant(full_index, dtype=tf.int64)
      sparse_y = tf.IndexedSlices(y, full_indices)
    return kv_var, tf_var, resource_var, sparse_y

  def check_optimizer(
      self,
      kv_var,
      tf_var,
      resource_var,
      sparse_y,
      sparse_opt,
      resource_opt,
      kv_sparse_opt,
  ):
    """Compare with tensorflow corresponding optimizer"""
    # Create grads
    sparse_grads_and_vars = [[sparse_y, tf_var]]
    resource_sparse_grads_and_vars = [[sparse_y, resource_var]]
    sparse_grads_and_kv_vars = [[sparse_y, kv_var]]
    kv_val = kv_var._read_variable_op()  # pylint: disable=protected-access
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    # Use a temp diretory to avoid the creation of a new summary file,
    # each time pytest runs.
    summary_dir = tempfile.mktemp()
    with tf.device("/cpu:0"):
      sparse_train_op = sparse_opt.apply_gradients(sparse_grads_and_vars)
      resource_train_op = resource_opt.apply_gradients(
          resource_sparse_grads_and_vars)
      kv_sparse_train_op = kv_sparse_opt.apply_gradients(
          sparse_grads_and_kv_vars)
      init_op = tf.compat.v1.global_variables_initializer()

      with self.session() as sess:
        graph = sess.graph
        train_writer = tf.compat.v1.summary.FileWriter(summary_dir)
        train_writer.add_graph(graph)
        sess.run(init_op)

        profiler = model_analyzer.Profiler(graph)
        sess.run(
            kv_sparse_train_op,
            options=run_options,
            run_metadata=run_metadata,
        )
        profiler.add_step(step=1, run_meta=run_metadata)
        kv_result = sess.run(kv_val)
        kv_result_dict = {}
        for k, v in zip(kv_result[0], kv_result[1]):
          kv_result_dict[k.item()] = v
        sess.run(
            sparse_train_op,
            options=run_options,
            run_metadata=run_metadata,
        )
        profiler.add_step(step=2, run_meta=run_metadata)
        sparse_result = sess.run(tf_var)
        sess.run(
            resource_train_op,
            options=run_options,
            run_metadata=run_metadata,
        )
        profiler.add_step(step=3, run_meta=run_metadata)
        resource_result = sess.run(resource_var)
        result = True
        result_for_tf = True
        for k, v in kv_result_dict.items():
          sparse_var = sparse_result[k, :]
          resource_var = resource_result[k, :]
          result_for_tf = (result_for_tf and (sparse_var == resource_var).all())
          result = (result and (sparse_var == v).all() and result_for_tf)

      opts = option_builder.ProfileOptionBuilder.time_and_memory()
      profiler.profile_operations(options=opts)
      return result, result_for_tf

  def check_optimizer_v3(
      self,
      kv_var,
      tf_var,
      resource_var,
      sparse_y,
      sparse_opt,
      resource_opt,
      kv_sparse_opt,
      atol=1e-8,
  ):
    """Compare with tensorflow corresponding optimizer"""
    # Create grads
    sparse_grads_and_vars = [[sparse_y, tf_var]]
    resource_sparse_grads_and_vars = [[sparse_y, resource_var]]
    sparse_grads_and_kv_vars = [[sparse_y, kv_var]]
    kv_val = kv_var._read_variable_op()  # pylint: disable=protected-access
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    # Use a temp diretory to avoid the creation of a new summary file,
    # each time pytest runs.
    summary_dir = tempfile.mktemp()
    with tf.device("/cpu:0"):
      sparse_train_op = sparse_opt.apply_gradients(sparse_grads_and_vars)
      resource_train_op = resource_opt.apply_gradients(
          resource_sparse_grads_and_vars)
      kv_sparse_train_op = kv_sparse_opt.apply_gradients(
          sparse_grads_and_kv_vars)
      init_op = tf.compat.v1.global_variables_initializer()

      with self.session() as sess:
        graph = sess.graph
        train_writer = tf.compat.v1.summary.FileWriter(summary_dir)
        train_writer.add_graph(graph)
        sess.run(init_op)

        profiler = model_analyzer.Profiler(graph)
        sess.run(
            kv_sparse_train_op,
            options=run_options,
            run_metadata=run_metadata,
        )
        profiler.add_step(step=1, run_meta=run_metadata)
        kv_result = sess.run(kv_val)
        # print('kv_result: {}'.format(kv_result))
        kv_result_dict = {}
        for k, v in zip(kv_result[0], kv_result[1]):
          kv_result_dict[k.item()] = v
        sess.run(
            sparse_train_op,
            options=run_options,
            run_metadata=run_metadata,
        )
        profiler.add_step(step=2, run_meta=run_metadata)
        sparse_result = sess.run(tf_var)
        # print('sparse_result: {}'.format(sparse_result))
        sess.run(
            resource_train_op,
            options=run_options,
            run_metadata=run_metadata,
        )
        profiler.add_step(step=3, run_meta=run_metadata)
        resource_result = sess.run(resource_var)
        # print('resource_result: {}'.format(resource_result))
        result = True
        result_for_tf = True
        for k, v in kv_result_dict.items():
          sparse_var = sparse_result[k, :]
          resource_var = resource_result[k, :]
          result_for_tf = (result_for_tf and (sparse_var == resource_var).all())
          result = (result and np.allclose(sparse_var, v, atol=atol)
                    and result_for_tf)

      opts = option_builder.ProfileOptionBuilder.time_and_memory()
      profiler.profile_operations(options=opts)
      return result, result_for_tf

  def test_adam_optimizer(self):
    """Test TFPlus adam optimizer"""
    kv_var, tf_var, resource_var, sparse_y = self.init_test_data(h=10)
    # Set initial learning rate for Adam optimizer
    learning_rate = 0.1

    # Use tf variable, resource variable and our kv_variable to check result correctness.
    sparse_opt = AdamOptimizer(learning_rate=learning_rate)
    kv_sparse_opt = AdamOptimizer(learning_rate=learning_rate)
    resource_sparse_opt = AdamOptimizer(learning_rate=learning_rate)

    result, result_for_tf = self.check_optimizer(
        kv_var,
        tf_var,
        resource_var,
        sparse_y,
        sparse_opt,
        resource_sparse_opt,
        kv_sparse_opt,
    )
    self.assertEqual(True, result)
    self.assertEqual(True, result_for_tf)

  def test_adagrad_optimizer(self):
    """Test adagrad for both kv variable and tf variable"""
    kv_var, tf_var, resource_var, sparse_y = self.init_test_data(h=10)
    # Use tf variable, resource variable and our kv_variable to check result correctness.
    sparse_opt = AdagradOptimizer(0.5)
    resource_sparse_opt = AdagradOptimizer(0.5)
    kv_sparse_opt = AdagradOptimizer(0.5)
    result, result_for_tf = self.check_optimizer_v3(
        kv_var,
        tf_var,
        resource_var,
        sparse_y,
        sparse_opt,
        resource_sparse_opt,
        kv_sparse_opt,
    )
    self.assertEqual(True, result_for_tf)
    self.assertEqual(True, result)

  def test_group_adam_v4_optimizer(self):
    """Test gradient adam for both kv variable and tf variable"""
    kv_var, tf_var, resource_var, sparse_y = self.init_test_data(h=10)
    # Use tf variable, resource variable and our kv_variable to check result correctness.
    sparse_opt = GroupAdamOptimizer(0.5, version=4)
    kv_sparse_opt = GroupAdamOptimizer(0.5, version=4)
    resource_sparse_opt = AdamOptimizer(0.5)  # just adam
    result, result_for_tf = self.check_optimizer_v3(
        kv_var,
        tf_var,
        resource_var,
        sparse_y,
        sparse_opt,
        resource_sparse_opt,
        kv_sparse_opt,
    )
    self.assertEqual(True, result)
    self.assertEqual(True, result_for_tf)

  def test_group_adam_v4_optimizer_with_1embedding_dim(self):
    """Test gradient adam for both kv variable and tf variable"""
    kv_var, tf_var, resource_var, sparse_y = self.init_test_data(h=10, w=1)
    # Use tf variable, resource variable and our kv_variable to check result correctness.
    sparse_opt = GroupAdamOptimizer(0.5, version=4)
    kv_sparse_opt = GroupAdamOptimizer(0.5, version=4)
    resource_sparse_opt = AdamOptimizer(0.5)  # just adam
    result, result_for_tf = self.check_optimizer_v3(
        kv_var,
        tf_var,
        resource_var,
        sparse_y,
        sparse_opt,
        resource_sparse_opt,
        kv_sparse_opt,
    )
    self.assertEqual(True, result)
    self.assertEqual(True, result_for_tf)

  def test_sparse_group_ftrl_optimizer(self):
    """Test sparse group ftrl for both kv variable and tf variable"""
    kv_var, tf_var, resource_var, sparse_y = self.init_test_data(h=10)
    # Use tf variable, resource variable and our kv_variable to check result correctness.
    sparse_opt = SparseGroupFtrlOptimizer(
        0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.05,
        l21_regularization_strength=0.05,
    )
    kv_sparse_opt = SparseGroupFtrlOptimizer(
        0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.05,
        l21_regularization_strength=0.05,
    )
    resource_sparse_opt = SparseGroupFtrlOptimizer(
        0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.05,
        l21_regularization_strength=0.05,
    )
    # The result will not be same, we just make sparse group lasso successful run.
    result, result_for_tf = self.check_optimizer(
        kv_var,
        tf_var,
        resource_var,
        sparse_y,
        sparse_opt,
        resource_sparse_opt,
        kv_sparse_opt,
    )
    self.assertEqual(False, result)
    self.assertEqual(True, result_for_tf)

  def test_sparse_group_ftrl_optimizer_with_1embedding_dim(self):
    """Test sparse group ftrl for both kv variable and tf variable"""
    kv_var, tf_var, resource_var, sparse_y = self.init_test_data(h=10, w=1)
    # Use tf variable, resource variable and our kv_variable to check result correctness.
    sparse_opt = SparseGroupFtrlOptimizer(
        0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.05,
        l21_regularization_strength=0.05,
    )
    kv_sparse_opt = SparseGroupFtrlOptimizer(
        0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.05,
        l21_regularization_strength=0.05,
    )
    resource_sparse_opt = SparseGroupFtrlOptimizer(
        0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.05,
        l21_regularization_strength=0.05,
    )
    # The result will not be same, we just make sparse group lasso successful run.
    result, result_for_tf = self.check_optimizer(
        kv_var,
        tf_var,
        resource_var,
        sparse_y,
        sparse_opt,
        resource_sparse_opt,
        kv_sparse_opt,
    )
    self.assertEqual(False, result)
    self.assertEqual(True, result_for_tf)


if __name__ == "__main__":
  test.main()
