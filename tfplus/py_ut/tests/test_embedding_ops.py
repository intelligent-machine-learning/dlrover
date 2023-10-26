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
"""Tests for kvVariable embedding ops."""
from __future__ import absolute_import, division, print_function

import ctypes
from functools import partial

import numpy as np
import tensorflow as tf

# import embedding_ops to hook original tensorflow embedding_ops
# pylint: disable=unused-import, wrong-import-order
# isort: off
from tfplus.kv_variable.python.ops import embedding_ops
from tfplus.kv_variable.python.ops.embedding_ops import (
    insert_kv_embedding,
    safe_embedding_lookup_sparse_v2,
)
from tfplus.kv_variable.python.ops.kv_variable_ops import gen_kv_variable_ops
from tfplus.kv_variable.python.ops.variable_scope import get_kv_variable
from tfplus.kv_variable.python.training import GradientDescentOptimizer
from tensorflow.python import keras
from tensorflow.python.framework import constant_op, dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import state_ops, variables
# pylint: disable=ungrouped-imports
from tensorflow.python.ops.embedding_ops import (
    embedding_lookup,
    embedding_lookup_sparse,
    safe_embedding_lookup_sparse,
)

tf.compat.v1.disable_eager_execution()

class KvVariableEmbeddingOpsTest(tf.test.TestCase):
  """This class implements a set of kvVariable embedding ops test functions"""

  def __init__(self, method_name="run_kv_variable_embedding_ops_test"):
    super(KvVariableEmbeddingOpsTest, self).__init__(methodName=method_name)

  # pylint: disable=missing-docstring
  def _ids_2d(self):
    max_shape = 10000
    indices = [[0, i] for i in range(100)]
    ids = list(range(100))
    shape = [1, max_shape]

    sparse_ids = sparse_tensor_lib.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int64),
        constant_op.constant(shape, dtypes.int64),
    )
    return sparse_ids

  def _invalid_ids_and_weights_2d(self):
    # Each row demonstrates a test case:
    #   Row 0: multiple valid ids, 1 invalid id, weighted mean
    #   Row 1: all ids are invalid (leaving no valid ids after pruning)
    #   Row 2: no ids to begin with
    #   Row 3: single id
    #   Row 4: all ids have <=0 weight
    indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
    ids = [0, 1, -1, -1, 2, 0, 1]
    weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
    shape = [5, 64]

    sparse_ids = sparse_tensor_lib.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int64),
        constant_op.constant(shape, dtypes.int64),
    )

    sparse_weights = sparse_tensor_lib.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(weights, dtypes.float32),
        constant_op.constant(shape, dtypes.int64),
    )

    return sparse_ids, sparse_weights

  # pylint: disable=missing-docstring
  def _const_weights(self, embedding_dim=64, num_shards=1, enter_threshold=0):
    assert embedding_dim > 0
    assert num_shards > 0

    # tf.Auto_REUSE for sharing the same variable
    with tf.compat.v1.variable_scope("test_kv", reuse=tf.compat.v1.AUTO_REUSE):
      embedding_weights = get_kv_variable(
          "kv_embedding",
          embedding_dim=embedding_dim,
          key_dtype=tf.int64,
          value_dtype=tf.float32,
          partitioner=tf.compat.v1.fixed_size_partitioner(
              num_shards=num_shards),
          initializer=tf.compat.v1.ones_initializer,
          enter_threshold=enter_threshold,
      )
    params = embedding_weights
    if isinstance(params, variables.PartitionedVariable):
      params = list(params)  # Iterate to get the underlying Variables.
    if not isinstance(params, list):
      params = [params]
    ids = list(range(100))
    fill_values = [[i] * embedding_dim for i in ids]
    p_assignments = [i % num_shards for i in ids]
    scatter_keys = []
    scatter_values = []
    for i in range(num_shards):
      scatter_keys.append([])
      scatter_values.append([])
    for i in ids:
      scatter_keys[p_assignments[i]].append(i)
      scatter_values[p_assignments[i]].append(fill_values[i])

    scatter_ops = [
        state_ops.scatter_update(
            params[i],
            constant_op.constant(scatter_keys[i], dtypes.int64),
            constant_op.constant(scatter_values[i], dtypes.float32),
        ) for i in range(num_shards)
    ]

    # get_count_ops = [
    #     params[i].get_counting(
    #         constant_op.constant(scatter_keys[i], dtypes.int64))
    #     for i in range(num_shards)
    # ]
    # get_count_ops = None
    return embedding_weights, scatter_ops, None

  def _construct_graph(self, num_shards=2, enter_threshold=1, is_training=True):
    if is_training:
      keras.backend.set_learning_phase(1)
    else:
      keras.backend.set_learning_phase(0)
    params, _, _ = self._const_weights(
        embedding_dim=64,
        num_shards=num_shards,
        enter_threshold=enter_threshold,
    )
    sparse_ids = self._ids_2d()
    embedding = embedding_lookup_sparse(params,
                                        sparse_ids,
                                        None,
                                        combiner="sum")
    return embedding, None

  # pylint: disable=missing-docstring
  def test_embedding_lookup(self):
    learning_phase = keras.backend.learning_phase()
    with tf.compat.v1.variable_scope("no_shards"):
      # num_shards=1, no partitioner
      params_no_shard, scatter_ops_no_shard, _ = self._const_weights(
          embedding_dim=64, num_shards=1)

    with tf.compat.v1.variable_scope("with_shards"):
      # num_shards=10, will return 10 parts of KvVariable,
      # embedding_lookup does support this case with mod partition_strategy
      (
          params_with_shards,
          scatter_ops_with_shards,
          _,
      ) = self._const_weights(embedding_dim=64, num_shards=10)

    ids = self._ids_2d().values
    embedding_lookup_1 = embedding_lookup(params_no_shard, ids)
    embedding_lookup_2 = embedding_lookup(params_with_shards, ids)
    global_init = tf.compat.v1.global_variables_initializer()

    with self.session() as sess:
      sess.run(global_init)
      # Lookup in inference mode, will return all zeros vector
      # TODO: support inference mode
      # lookup_result_1, lookup_result_2 = sess.run(
      #     [embedding_lookup_1, embedding_lookup_2],
      #     feed_dict={learning_phase.name: False})
      # self.assertAllEqual(lookup_result_1.shape, (100, 64))
      # self.assertAllEqual(lookup_result_2.shape, (100, 64))
      # self.assertTrue((lookup_result_1 == 0.0).all())
      # self.assertTrue((lookup_result_2 == 0.0).all())

      # Lookup in training mode, will return all ones vector
      lookup_result_1, lookup_result_2 = sess.run(
          [embedding_lookup_1, embedding_lookup_2],
          feed_dict={learning_phase.name: True},
      )
      self.assertAllEqual(lookup_result_1.shape, (100, 64))
      self.assertAllEqual(lookup_result_2.shape, (100, 64))
      self.assertTrue((lookup_result_1 == 1.0).all())
      self.assertTrue((lookup_result_2 == 1.0).all())
      # Scatter ops, each id will be set vector with [id] * 64
      sess.run([scatter_ops_no_shard, scatter_ops_with_shards])
      lookup_result_1, lookup_result_2 = sess.run(
          [embedding_lookup_1, embedding_lookup_2],
          feed_dict={learning_phase.name: True},
      )
      self.assertAllEqual(lookup_result_1.shape, (100, 64))
      self.assertAllEqual(lookup_result_2.shape, (100, 64))
      result_value = [[i] * 64 for i in range(100)]
      self.assertAllEqual(lookup_result_1, result_value)
      self.assertAllEqual(lookup_result_2, result_value)

  # pylint: disable=missing-docstring
  def test_embedding_lookup_use_const_learning_phase(self):
    with tf.compat.v1.variable_scope("no_shards"):
      # num_shards=1, no partitioner
      params_no_shard, scatter_ops_no_shard, _ = self._const_weights(
          embedding_dim=64, num_shards=1)

    with tf.compat.v1.variable_scope("with_shards"):
      # num_shards=10, will return 10 parts of KvVariable,
      # embedding_lookup does support this case with mod partition_strategy
      (
          params_with_shards,
          scatter_ops_with_shards,
          _,
      ) = self._const_weights(embedding_dim=64, num_shards=2)

    ids = self._ids_2d().values

    def lookup_assert():
      embedding_lookup_1 = embedding_lookup(params_no_shard, ids)
      embedding_lookup_2 = embedding_lookup(params_with_shards, ids)
      global_init = tf.compat.v1.global_variables_initializer()

      with self.session() as sess:
        sess.run(global_init)
        # Lookup in training mode, will return all ones vector
        lookup_result_1, lookup_result_2 = sess.run(
            [embedding_lookup_1, embedding_lookup_2])
        # self.assertAllEqual(lookup_result_1.shape, (100, 64))
        # self.assertAllEqual(lookup_result_2.shape, (100, 64))
        self.assertTrue((lookup_result_1 == 1.0).all())
        self.assertTrue((lookup_result_1 == 1.0).all())
        # Scatter ops, each id will be set vector with [id] * 64
        sess.run([scatter_ops_no_shard, scatter_ops_with_shards])
        lookup_result_1, lookup_result_2 = sess.run(
            [embedding_lookup_1, embedding_lookup_2])
        # self.assertAllEqual(lookup_result_1.shape, (100, 64))
        # self.assertAllEqual(lookup_result_2.shape, (100, 64))
        result_value = [[i] * 64 for i in range(100)]

        self.assertAllEqual(lookup_result_1, result_value)
        self.assertAllEqual(lookup_result_2, result_value)

    lookup_assert()

  def test_embedding_lookup_sparse(self):
    learning_phase = keras.backend.learning_phase()
    # num_shards=10, will return 10 parts of KvVariable,
    # embedding_lookup_sparse does support this case with mod partition_strategy
    params, scatter_ops, _ = self._const_weights(embedding_dim=64,
                                                 num_shards=10)
    sparse_ids = self._ids_2d()
    embedding_1 = embedding_lookup_sparse(params,
                                          sparse_ids,
                                          None,
                                          combiner="sum")
    embedding_2 = embedding_lookup_sparse(params,
                                          sparse_ids,
                                          None,
                                          combiner="mean")

    global_init = tf.compat.v1.global_variables_initializer()
    with self.session() as sess:
      sess.run(global_init)
      # Lookup in training mode
      embedding_result_1, embedding_result_2 = sess.run(
          [embedding_1, embedding_2],
          feed_dict={learning_phase.name: True},
      )
      self.assertAllEqual(embedding_result_1.shape, (1, 64))
      self.assertAllEqual(embedding_result_2.shape, (1, 64))
      self.assertTrue((embedding_result_1 == 100.0).all())
      self.assertTrue((embedding_result_2 == 1.0).all())
      # Scatter ops
      sess.run(scatter_ops)
      embedding_result_1, embedding_result_2 = sess.run(
          [embedding_1, embedding_2],
          feed_dict={learning_phase.name: True},
      )
      self.assertAllEqual(embedding_result_1.shape, (1, 64))
      self.assertAllEqual(embedding_result_2.shape, (1, 64))
      sum_val = sum(i for i in range(100))
      mean_val = sum_val / 100
      self.assertTrue((embedding_result_1 == sum_val).all())
      self.assertTrue((embedding_result_2 == mean_val).all())

  # pylint: disable=missing-docstring
  def test_safe_embedding_lookup_sparse(self):
    # num_shards=10, will return 10 parts of KvVariable,
    params, scatter_ops, _ = self._const_weights(embedding_dim=64,
                                                 num_shards=10)
    sparse_ids, _ = self._invalid_ids_and_weights_2d()
    safe_embedding_lookup_result = safe_embedding_lookup_sparse(params,
                                                                sparse_ids,
                                                                None,
                                                                combiner="mean")
    safe_embedding_lookup_result_v2 = safe_embedding_lookup_sparse_v2(
        params, sparse_ids, None, combiner="mean")
    indices = tf.convert_to_tensor([0, 1, 2, -1], dtype=tf.int64)
    embedding_weights = embedding_lookup(params, indices)
    global_init = tf.compat.v1.global_variables_initializer()
    with self.session() as sess:
      sess.run(global_init)
      sess.run(scatter_ops)
      # pylint: disable=line-too-long
      (
          safe_embedding_lookup_result,
          safe_embedding_lookup_result_v2,
          embedding_weights,
      ) = sess.run([
          safe_embedding_lookup_result,
          safe_embedding_lookup_result_v2,
          embedding_weights,
      ])
      except_result = np.array([
          (embedding_weights[0] + embedding_weights[1] + embedding_weights[3]) /
          3.0,
          embedding_weights[3],
          [0] * 64,
          embedding_weights[2],
          (embedding_weights[0] + embedding_weights[1]) / 2.0,
      ])
      self.assertAllClose(safe_embedding_lookup_result, except_result)
      self.assertAllClose(safe_embedding_lookup_result_v2, except_result)

  def test_embedding_lookup_sparse_with_counting(self):
    learning_phase = keras.backend.learning_phase()
    params, _, _ = self._const_weights(embedding_dim=64,
                                               num_shards=2,
                                               enter_threshold=1)
    sparse_ids = self._ids_2d()
    embedding = embedding_lookup_sparse(params,
                                        sparse_ids,
                                        None,
                                        combiner="sum")
    loss = tf.reduce_sum(tf.add(embedding, 1.0, name="add"))
    optimzier = GradientDescentOptimizer(1)
    gradients = optimzier.compute_gradients(loss)
    train_op = optimzier.apply_gradients(gradients)
    global_init = tf.compat.v1.global_variables_initializer()
    with self.session() as sess:
      sess.run(global_init)
      # Lookup in inference mode
      # embedding_result = sess.run(embedding)
      # self.assertAllEqual(embedding_result.shape, (1, 64))
      # self.assertTrue((embedding_result == 0.0).all())

      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 0).all())
      # Lookup in training mode
      embedding_result = sess.run(embedding,
                                  feed_dict={learning_phase.name: True})
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 1).all())

      embedding_result = sess.run(embedding)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 1).all())

      # Lookup in training mode
      embedding_result = sess.run(embedding,
                                  feed_dict={learning_phase.name: True})
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 2).all())
      sess.run(train_op, feed_dict={learning_phase.name: True})

  def test_embedding_lookup_sparse_with_counting_with_different_graph(self):
    embedding_train_ops, _ = self._construct_graph()
    embedding_predict_ops, _ = self._construct_graph(is_training=False)
    global_init = tf.compat.v1.global_variables_initializer()
    with self.session() as sess:
      # tf.io.write_graph(sess.graph_def, './my-model', 'train.pbtxt')
      sess.run(global_init)
      # embedding_result = sess.run(embedding_predict_ops)
      # self.assertAllEqual(embedding_result.shape, (1, 64))
      # self.assertTrue((embedding_result == 0.0).all())

      # count_result = sess.run(count_ops)
      # for count in count_result:
      # self.assertTrue((count == 0).all())

      embedding_result = sess.run(embedding_train_ops)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 1).all())

      embedding_result = sess.run(embedding_predict_ops)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 1).all())

      embedding_result = sess.run(embedding_train_ops)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 2).all())

  # pylint: disable=pointless-string-statement
  def test_embedding_lookup_sparse_with_no_counting(self):
    # Set enter_threshold to zero, will not increase counting
    embedding_train_ops, _ = self._construct_graph(enter_threshold=0)
    embedding_predict_ops, _ = self._construct_graph(is_training=False,
                                                     enter_threshold=0)
    global_init = tf.compat.v1.global_variables_initializer()
    with self.session() as sess:
      sess.run(global_init)
      # embedding_result = sess.run(embedding_predict_ops)
      # self.assertAllEqual(embedding_result.shape, (1, 64))
      # self.assertTrue((embedding_result == 0.0).all())

      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 0).all())

      embedding_result = sess.run(embedding_train_ops)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 1).all())

      embedding_result = sess.run(embedding_predict_ops)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 1).all())

      embedding_result = sess.run(embedding_train_ops)
      self.assertAllEqual(embedding_result.shape, (1, 64))
      self.assertTrue((embedding_result == 100.0).all())
      # count_result = sess.run(count_ops)
      # for count in count_result:
      #   self.assertTrue((count == 2).all())

  def test_insert_kv_embedding_via_pointer(self):
    # This op is only use for grappler in tf runtime, never use in
    # tf python api EXCEPT test.  Note `kv_variable_gather_or_insert_pointer'
    # not controlled by learning phase
    if not hasattr(gen_kv_variable_ops, "kv_variable_gather_or_insert_pointer"):
      return

    def generate_test(kv_size, kv_dims, rank_num):
      # generate id and kv pairs
      # ground truth ops origanized
      # [[lookup(kv_i, id_ij) for i in range(kv_size)] for _ in range(rank_num)]
      # ids are [[id_ij for i in range(kv_size)] for _ in range(rank_num)]
      # concat id is [[size(id_ij) for for i in range(kv_size)] + [id_ij) for for i in range(kv_size)] for j in range(rank_num)]
      insert_via_pointer = (
          gen_kv_variable_ops.kv_variable_gather_or_insert_pointer)
      gen_kv = partial(get_kv_variable,
                       key_dtype=tf.int64,
                       value_dtype=tf.float32)
      gen_id = partial(tf.constant, dtype=tf.int64)
      kvs = [
          gen_kv(str(i), embedding_dim=dim)
          for i, dim in zip(range(kv_size), kv_dims)
      ]
      ids = []
      concat_ids = []
      gt_ops = []
      for _ in range(rank_num):
        ids.append([
            np.random.randint(0, 10240, np.random.randint(1024, 2048, 1))
            for _ in range(kv_size)
        ])

        for index, j in enumerate(ids[-1]):
          concat_ids.append(j.size)
          gt_ops.append(
              gen_kv_variable_ops.kv_variable_gather_or_insert_v2(
                  kvs[index].handle, gen_id(j), dtype=tf.float32))
        for j in ids[-1]:
          concat_ids.extend(j)
      batched_e_meta = insert_via_pointer(
          [i.handle for i in kvs],
          gen_id(concat_ids),
          dtype=tf.float32,
          rank_num=rank_num,
          has_embedding_meta=True,
      )
      batched_e = insert_via_pointer(
          [i.handle for i in kvs],
          gen_id(concat_ids),
          dtype=tf.float32,
          rank_num=rank_num,
          has_embedding_meta=False,
      )
      return batched_e_meta, batched_e, gt_ops, ids

    id_size = kv_size = 10
    rank_num = 5
    dims = np.random.randint(1, 10, 10)
    batched_e_meta, batched_e, gt, gt_id = generate_test(
        kv_size, dims, rank_num)
    global_init = tf.compat.v1.global_variables_initializer()
    with self.session() as sess:
      sess.run(global_init)
      # pylint: disable=unused-variable
      (
          emb_no_meta,
          emb_part_no_meta,
          emb_part_each_no_meta,
          indices_no_meta,
      ) = sess.run(batched_e)
      # pylint: enable=unused-variable
      (
          emb_meta,
          emb_part_meta,
          emb_part_each_meta,
          indices_meta,
      ) = sess.run(batched_e_meta)
      emb_gt = sess.run(gt)
      gt_id_result = [[] for _ in range(kv_size)]
      for ids in gt_id:
        for index, i in enumerate(ids):
          gt_id_result[index].append(i)
      gt_id_result_concat = [
          np.concatenate(gt_id_result[i], 0) for i in range(kv_size)
      ]
      # test lookup id
      for index, (i, j) in enumerate(zip(gt_id_result_concat, indices_no_meta)):
        self.assertAllEqual(i, j)
        self.assertAllEqual(j, indices_meta[index])
      # header only in int32, maybe cause memory mis aligment
      header_array = emb_meta.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
      emb_cursor_meta = 0
      emb_cursor_without_meta = 0
      header_cursor = 0
      index = 0
      self.assertAllEqual(emb_part_no_meta, emb_part_meta)
      for rank in range(rank_num):
        emb_cursor_meta += id_size
        for feature_index, dim in enumerate(dims):
          # test for emb with headers
          size = header_array[header_cursor]
          emb_meta_v = emb_meta[emb_cursor_meta:emb_cursor_meta + size].reshape(
              -1, dim)
          self.assertAllEqual(emb_meta_v, emb_gt[index])

          # test for emb without headers
          size_from_parts = emb_part_each_meta[rank, feature_index]
          emb_no_meta_v = emb_no_meta[
              emb_cursor_without_meta:emb_cursor_without_meta +
              size_from_parts].reshape(-1, dim)
          self.assertAllEqual(emb_no_meta_v, emb_gt[index])

          # update offset
          emb_cursor_meta += size
          emb_cursor_without_meta += size_from_parts
          header_cursor += 1
          index += 1
        header_cursor = emb_cursor_meta


if __name__ == "__main__":
  tf.test.main()
