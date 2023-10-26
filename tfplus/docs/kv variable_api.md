## Kv Variable
<div id="top" align="center">

   | [English](./kv%20variable_api.md) | [中文](./kv%20variable_api_ZH.md) |

</div>

We have implemented an entirely new Variable, named KvVariable.
Its internal data type utilizes key-value format for storage (such as using a hash map structure),
and supports the following features:
 * KvVariable only need embedding_dimension to be created. New features will be dynamically added to the variable
 * KvVariable is primarily used for embedding calculations, it support forward and backward computations of embedding_lookup, and support parameter updates of all optimizers.
 * KvVariable supports PartitionedVariable for sharding large embeddings.
 * KvVariable supports save/restore with checkpoint, savedmodel formats, and supports automatic clipping of sparse vectors when exported.
 
## User Interface

   The tf.get_kv_variable is a low-level API supporting tf variable_scope and partitioner.
   ```python
      def get_kv_variable(name,
                          embedding_dim=None,
                          key_dtype=dtypes.int64,
                          value_dtype=dtypes.float32,
                          initializer=None,
                          trainable=None,
                          collections=None,
                          partitioner=None,
                          constraint=None):
   ```
* name: name of the variable.
* embedding_dim: dimension of the variable.
* key_dtype: type of the variable key (default is tf.int64; accept int, int32).
* value_dtype: type of the variable value (default is tf.float32).
* initializer: Initializer or Tensor. If it is a Tensor, its shape must be defined.
* trainable: a collection list that the variable will be added to, default is [GraphKeys.GLOBAL_VARIABLES] (see tf.Variable).
* partitioner: optional parameter, accepts a partition function, features in the variable will be evenly distributed into several partitions.

   tf.get_kv_variable low level api, with tf.variable_scope and partitioner support
   ```python
      def get_kv_variable(name,
                          embedding_dim=None,
                          key_dtype=dtypes.int64,
                          value_dtype=dtypes.float32,
                          initializer=None,
                          trainable=None,
                          collections=None,
                          partitioner=None,
                          constraint=None):
   ```
embedding ops：Embedding ops: support the following three types of lookup. Same with the original TensorFlow interfaces.
   * [tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
   * [tf.nn.embedding_lookup_sparse](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse)
   * [tf.nn.safe_embedding_lookup_sparse](https://www.tensorflow.org/api_docs/python/tf/nn/safe_embedding_lookup_sparse)

## low level api example
   ```python
   import tensorflow as tf
   import os

   ckp_dir = './checkpoint'
   ckp_path = os.path.join(ckp_dir, 'model.ckpt')

   num_shards = 5
   with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
   var = tf.get_kv_variable("kv_embedding",
                              embedding_dim=64,
                              key_dtype=tf.int64,
                              initializer=tf.compat.v1.ones_initializer(),
                           partitioner=tf.compat.v1.fixed_size_partitioner(num_shards=num_shards))

   emb = tf.nn.embedding_lookup(var, tf.convert_to_tensor([1, 2, 3,6,8,9], dtype=tf.int64))
   emb1 = tf.nn.embedding_lookup(var, tf.convert_to_tensor([1000000000000000], dtype=tf.int64))
   fun = tf.add(emb, 1.0, name='add')
   loss = tf.reduce_sum(fun)
   opt = tf.train.FtrlOptimizer(0.005,
                              l1_regularization_strength=0.025,
                              l2_regularization_strength=0.00001)
   g_v = opt.compute_gradients(loss)
   train_op = opt.apply_gradients(g_v)
   init = tf.global_variables_initializer()
   saver = tf.train.Saver()

   with tf.Session() as sess:
   sess.run(init)
   print(sess.run({'emb':emb, 'fun':fun, 'train': train_op}))
   print(sess.run(emb1))
   save_path = saver.save(sess, ckp_path)
   print'model saved in file %s' % save_path)

   with tf.Session() as sess:
   saver.restore(sess, ckp_path)
   print(sess.run(emb1))
   print(sess.run({'emb':emb, 'fun':fun}))
   print(sess.run(emb1))
   ```
