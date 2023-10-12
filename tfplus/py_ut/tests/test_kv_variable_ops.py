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
"""Tests for kv_variable ops."""

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

from tfplus.kv_variable.python.ops.kv_variable_ops import \
    gen_kv_variable_ops as gen_kv_var_ops

tf.compat.v1.disable_eager_execution()

kv_variable_v2 = gen_kv_var_ops.kv_variable
kv_variable_shape = gen_kv_var_ops.kv_variable_shape_v2
init_kv_variable_v2 = gen_kv_var_ops.init_kv_variable_v2
kv_variable_is_initialized_v2 = gen_kv_var_ops.kv_variable_is_initialized_v2
kv_variable_size_v2 = gen_kv_var_ops.kv_variable_size_v2
kv_variable_frequency = gen_kv_var_ops.kv_variable_frequency
read_kv_variable_op_v2 = gen_kv_var_ops.read_kv_variable_op_v2
destroy_kv_variable_op_v2 = gen_kv_var_ops.destroy_kv_variable_op_v2
kv_variable_gather_v2 = gen_kv_var_ops.kv_variable_gather_or_zeros_v2
kv_variable_gather_or_insert_v2 = (
    gen_kv_var_ops.kv_variable_gather_or_insert_v2)
kv_variable_insert_v2 = gen_kv_var_ops.kv_variable_insert_v2
kv_variable_inc_count_v2 = gen_kv_var_ops.kv_variable_increase_count_v2
kv_variable_export_v2 = gen_kv_var_ops.kv_variable_export
kv_variable_import_v2 = gen_kv_var_ops.kv_variable_import
kv_variable_delete = gen_kv_var_ops.kv_variable_delete
kv_variable_get_time_stamp = gen_kv_var_ops.kv_variable_get_time_stamp
kv_variable_delete_with_timestamp = (
    gen_kv_var_ops.kv_variable_delete_with_timestamp)


class KvVariableOpsTest(tf.test.TestCase):
  """kv variable test class"""

  def __init__(self, method_name="run_kv_variable_ops_test"):
    super(KvVariableOpsTest, self).__init__(methodName=method_name)

    # Create a kv variable
    self.container = ""
    self.shared_name = ""
    self.use_node_name_sharing = False
    self.enter_threshold = 0
    self.embedding_dim = 8
    self.key_dtype = tf.int64
    self.value_dtype = tf.float32

    # Number of rows of the initializing table
    self.init_table_rows = 1024

  def test_kv_variable_v2(self):
    """test kv variable creation"""
    # Create a kv varaible
    with self.session() as sess:
      var_handle = kv_variable_v2(
          container=self.container,
          shared_name=self.shared_name,
          use_node_name_sharing=self.use_node_name_sharing,
          key_dtype=self.key_dtype,
          value_dtype=self.value_dtype,
          value_shape=[self.embedding_dim],
          enter_threshold=self.enter_threshold,
      )
      # Evaluate the handle value
      sess.run(var_handle)

  def test_kv_variable_shape(self):
    """test kv variable shape"""

    # Create a kv variable
    var_handle = kv_variable_v2(
        container=self.container,
        shared_name=self.shared_name,
        use_node_name_sharing=self.use_node_name_sharing,
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
        enter_threshold=self.enter_threshold,
    )

    var_shape_op = kv_variable_shape(var_handle)
    with self.session() as sess:
      result = sess.run(var_shape_op)
      self.assertAllEqual(result, [0, self.embedding_dim])

  def test_kv_variable_is_initialized_v2(self):
    """test kv variable initializing status"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Initialize kv variable
    init_table = tf.compat.v1.random_normal(
        [self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    # Check if the variable is already initialized
    is_initialized_op = kv_variable_is_initialized_v2(var_handle)

    with self.session() as sess:
      # Not initialized
      is_initialized = sess.run(is_initialized_op)
      self.assertAllEqual([is_initialized], [False])

      # Initialized
      sess.run(init_var_op)
      is_initialized = sess.run(is_initialized_op)
      self.assertAllEqual([is_initialized], [True])

  def test_kv_variable_size_v2(self):
    """test to get kv variable size"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Initialize kv variable
    init_table = tf.compat.v1.random_normal(
        [self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    # Variable size op
    var_size_op = kv_variable_size_v2(var_handle, T=tf.int64)

    with self.session() as sess:
      # Initialize the table
      sess.run(init_var_op)

      # Get the number of elements in the variable
      var_size = sess.run(var_size_op)
      self.assertAllEqual([var_size], [0])

  def test_kv_variable_frequency(self):
    """test to get kv variable frequency"""
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
        enter_threshold=2,
    )
    init_table = tf.compat.v1.random_normal(
        [self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)
    var_freq_op = kv_variable_frequency(var_handle,
                                        name=var_handle.op.name + "/Freq")

    indices_0 = tf.compat.v1.convert_to_tensor([0, 1, 2, 3, 4],
                                               dtype=self.key_dtype)
    indices_1 = tf.compat.v1.convert_to_tensor([2, 3, 4, 5, 6],
                                               dtype=self.key_dtype)

    gather_training_0 = kv_variable_gather_or_insert_v2(var_handle,
                                                        indices=indices_0,
                                                        dtype=self.value_dtype)
    gather_training_1 = kv_variable_gather_or_insert_v2(var_handle,
                                                        indices=indices_1,
                                                        dtype=self.value_dtype)
    gather_0 = kv_variable_gather_v2(var_handle,
                                     indices=indices_0,
                                     dtype=self.value_dtype)

    with self.session() as sess:
      sess.run(init_var_op)
      sess.run(gather_training_0)
      print(sess.run(var_freq_op))
      self.assertAllEqual(sess.run(var_freq_op), 0)
      sess.run(gather_training_1)
      print(sess.run(var_freq_op))
      self.assertAllEqual(sess.run(var_freq_op), 6)
      sess.run(gather_0)
      print(sess.run(var_freq_op))
      self.assertAllEqual(sess.run(var_freq_op), 6)

  def test_destroy_kv_variable_op_v2(self):
    """test to destroy a kv variable instance"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Destroy the variable op
    destroy_var_op = destroy_kv_variable_op_v2(var_handle)

    with self.session() as sess:
      sess.run(destroy_var_op)

  def test_kv_variable_insert_v2(self):
    """test to insert key-value pairs into a kv variable"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Initialize kv variable
    init_table = tf.compat.v1.random_normal(
        [self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    # Insert key-value pairs into the variables
    vec_value = tf.compat.v1.convert_to_tensor(
        [1.0 for x in range(0, self.embedding_dim)], dtype=self.value_dtype)
    indices = tf.compat.v1.convert_to_tensor([0, 1, 2, 3, 4],
                                             dtype=self.key_dtype)
    values = tf.compat.v1.convert_to_tensor(
        [vec_value * x for x in range(0, 5)], dtype=self.value_dtype)
    insert_op = kv_variable_insert_v2(var_handle,
                                      indices=indices,
                                      values=values)
    with self.session() as sess:
      sess.run(init_var_op)
      sess.run(insert_op)

  def test_kv_variable_gather_v2(self):
    """test to gather values by keys from a kv varaible"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Initialize kv variable, use tf.ones
    init_table = tf.compat.v1.ones([self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    indices = tf.compat.v1.convert_to_tensor([0, 1, 2, 3, 4],
                                             dtype=self.key_dtype)
    # Gather indices do not use init value, except return all 0
    gather_for_inference = kv_variable_gather_v2(var_handle,
                                                 indices=indices,
                                                 dtype=self.value_dtype)

    # Gather the same indices use init value, except return all one
    gather_for_training = kv_variable_gather_or_insert_v2(
        var_handle, indices=indices, dtype=self.value_dtype)

    with self.session() as sess:
      sess.run(init_var_op)
      values = sess.run(gather_for_inference)
      self.assertAllEqual(values, np.zeros([5, self.embedding_dim],
                                           dtype=float))
      self.assertAllEqual(values.shape, (5, self.embedding_dim))
      print("gather res, ", values)

      values = sess.run(gather_for_training)
      self.assertAllEqual(values, np.ones([5, self.embedding_dim], dtype=float))
      self.assertAllEqual(values.shape, (5, self.embedding_dim))

  def test_kv_variable_gather_v2_with_random_init(self):
    """test to gather values by keys from a kv varaible"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Initialize kv variable, use random_uniform to generate
    # value in range[0.01, 1].
    init_table = tf.compat.v1.random_uniform(
        [self.init_table_rows, self.embedding_dim], minval=0.01, maxval=1.0)
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    indices = tf.compat.v1.convert_to_tensor([0, 1, 2, 3, 4],
                                             dtype=self.key_dtype)
    # Gather indices do not use init value, except return all 0
    gather_for_inference = kv_variable_gather_v2(var_handle,
                                                 indices=indices,
                                                 dtype=self.value_dtype)

    # Gather the same indices use init value,
    # except return in range [0.01, 1.0]
    gather_for_training = kv_variable_gather_or_insert_v2(
        var_handle, indices=indices, dtype=self.value_dtype)

    with self.session() as sess:
      sess.run(init_var_op)
      values = sess.run(gather_for_inference)
      self.assertAllEqual(values, np.zeros([5, self.embedding_dim],
                                           dtype=float))
      self.assertAllEqual(values.shape, (5, self.embedding_dim))

      values = sess.run(gather_for_training)
      # All value in values must also in range [0.01, 1.0]
      self.assertTrue((values >= 0.01).all())
      self.assertTrue((values <= 1.0).all())
      self.assertAllEqual(values.shape, (5, self.embedding_dim))

  def test_kv_variable_increase_count_v2(self):
    """test increase the count of keys in a kv variable"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
    )

    # Initialize kv variable
    init_table = tf.compat.v1.random_normal(
        [self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    # Insert key-value pairs into the variables
    vec_value = tf.compat.v1.convert_to_tensor(
        [1.0 for x in range(0, self.embedding_dim)], dtype=self.value_dtype)
    indices = tf.compat.v1.convert_to_tensor([0, 1, 2, 3, 4],
                                             dtype=self.key_dtype)
    values = tf.compat.v1.convert_to_tensor(
        [vec_value * x for x in range(0, 5)], dtype=self.value_dtype)
    insert_op = kv_variable_insert_v2(var_handle,
                                      indices=indices,
                                      values=values)

    # Increase counts
    counts = tf.compat.v1.convert_to_tensor([1, 2, 3, 4, 5], dtype=tf.int32)
    increase_count_op = kv_variable_inc_count_v2(var_handle,
                                                 indices=indices,
                                                 counts=counts)
    with self.session() as sess:
      sess.run(init_var_op)
      sess.run(insert_op)
      sess.run(increase_count_op)

  def test_kv_variable_import_v2(self):
    """test importing tensor values into kv variable"""
    # Create a kv variable
    var_handle = kv_variable_v2(
        key_dtype=self.key_dtype,
        value_dtype=self.value_dtype,
        value_shape=[self.embedding_dim],
        enter_threshold=1,
    )

    # Initialize kv variable
    init_table = tf.compat.v1.random_normal(
        [self.init_table_rows, self.embedding_dim])
    init_var_op = init_kv_variable_v2(var_handle, init_table)

    # Insert key-value pairs into the variables
    vec_value = tf.compat.v1.convert_to_tensor([1.0] * self.embedding_dim,
                                               dtype=self.value_dtype)
    indices = tf.compat.v1.convert_to_tensor([0, 1, 2, 3, 4],
                                             dtype=self.key_dtype)
    values = tf.compat.v1.convert_to_tensor(
        [vec_value * x for x in range(0, 5)], dtype=self.value_dtype)

    # Blacklist
    blacklist = tf.compat.v1.convert_to_tensor([7], dtype=self.key_dtype)

    # Frequency table
    freq_keys = tf.compat.v1.convert_to_tensor([1, 2, 3, 4, 5],
                                               dtype=self.key_dtype)
    freq_values = tf.compat.v1.convert_to_tensor([1, 2, 3, 4, 5],
                                                 dtype=tf.uint16)

    # Import ops
    import_ops = []
    for first_n in [3, 4, 6]:
      import_ops.append(
          kv_variable_import_v2(
              var_handle,
              keys=indices,
              values=values,
              init_table=init_table,
              blacklist=blacklist,
              freq_keys=freq_keys,
              freq_values=freq_values,
              first_n=first_n,
          ))

    # Export op
    export_op = kv_variable_export_v2(
        var_handle,
        Tkeys=self.key_dtype,
        Tvalues=self.value_dtype,
        first_n=6,
    )

    # Session
    with self.session() as sess:
      sess.run(init_var_op)
      # run export op
      for i in [0, 1, 2]:
        # import tensors
        sess.run(import_ops[i])

        # run export
        output = sess.run(export_op)
        self.assertTrue(len(output) == 6)

        exp_shapes = [[5], [5, self.embedding_dim]]
        if i == 0:
          exp_shapes.extend([
              [self.init_table_rows, self.embedding_dim],
              [0],
              [5],
              [5],
          ])
        elif i == 1:
          exp_shapes.extend([
              [self.init_table_rows, self.embedding_dim],
              [1],
              [6],
              [6],
          ])
        else:
          exp_shapes.extend([
              [self.init_table_rows, self.embedding_dim],
              [1],
              [6],
              [6],
          ])
        for j in [0, 1, 2, 3, 4, 5]:
          self.assertAllEqual(output[j].shape, exp_shapes[j])


if __name__ == "__main__":
  test.main()
