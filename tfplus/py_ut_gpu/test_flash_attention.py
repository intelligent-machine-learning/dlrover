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
"""Tests for flash attention ops."""
import math

import tensorflow as tf
from tensorflow.python.platform import test

from tfplus.flash_attn.python.ops.flash_attn_ops import FlashAttentionLayer

batch_size = 2
num_heads = 2
seq_len_q = 4
seq_len_v = 4
head_dim = 32
dtype = tf.half


def origin_attention(q, k, v):
  scores = tf.einsum('bthd,bshd->bhts', q / math.sqrt(q.shape[-1]), k)
  attention = tf.nn.softmax(scores)
  return tf.einsum('bhts,bshd->bthd', attention, v)


class FlashAttentionTest(test.TestCase):
  """flash attention test class"""

  def __init__(self, method_name='FlashAttentionTest'):
    super().__init__(method_name)

  def test_flash_attention(self):
    FlashAttentionTest._run(self)

  @staticmethod
  def _run(instance):
    """flash attention test func"""
    with tf.device("/gpu:0"):
      # Create input tensors
      queries = tf.random.normal(
          (batch_size, seq_len_q, num_heads, head_dim), dtype=dtype)
      keys = tf.random.normal(
          (batch_size, seq_len_v, num_heads, head_dim), dtype=dtype)
      values = tf.random.normal(
          (batch_size, seq_len_v, num_heads, head_dim), dtype=dtype)
      attention_output = origin_attention(queries, keys, values)

      flash_attention_layer = FlashAttentionLayer(
          seq_len_q, seq_len_v, num_heads, head_dim)
      flash_attention_output = flash_attention_layer(
          queries, keys, values)

      instance.assertTrue(
          (tf.reduce_max(
              tf.abs(
                  flash_attention_output - attention_output)) <= 1e-3))


if __name__ == '__main__':
  test.main()
