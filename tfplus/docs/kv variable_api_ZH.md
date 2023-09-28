## Kv Variable
<div id="top" align="center">

   | [English](./kv%20variable_api.md) | [中文](./kv%20variable_api_ZH.md) |

</div>

我们实现了一个全新的 Variable，取名 KvVariable。其内部数据类型采用 key-value 格式存储 (比如使用 hash map 结构),
并支持以下功能：
 * KvVariable 不需预先设定 dim0, 只需要 embedding_dimension，新出现的特征会被动态加入到训练中，并支持频次过滤；
 * KvVariable 主要用于 embedding 的计算，它能支持 embedding_lookup 的前向、反向计算，能支持所有优化器进行参数更新；
 * KvVariable 支持 PartitionedVariable 来进行对超大的 embedding 进行 shard，partition_strategy 使用 mod, 对于 string 类型，先 hash，后 mod；
 * KvVariable 支持模型 save、restore，兼容 checkpoint、savedmodel 格式，支持导出时自动裁剪稀疏向量。

## 用户接口

   tf.get_kv_variable low level api, 支持 tf variable_scope 和 partitioner
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
* name: variable 的名称
* embedding_dim: variable 的维度
* key_dtype: variable 的键类型（默认为 tf.int64，也可以接受 int、int32）
* value_dtype: variable 的值类型（默认为 tf.float32）
* initializer: Initializer 或 Tensor。如果是 Tensor，必须定义其形状
* trainable: 如果为 True，将 variable 添加到图集合 GraphKeys。TRAINABLE_VARIABLES 中参与优化器更新（参见 tf.Variable）
* collections: 一个 collection 列表，variable 会被加入到其中，默认为 [GraphKeys.GLOBAL_VARIABLES] (参见 tf.Variable)
* partitioner: 可选参数，接受一个分区函数，variable 中的特征会均匀分配在几个分区中

   tf.get_kv_variable low level api, 支持 tf variable_scope 和 partitioner
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
embedding ops：支持以下三种 lookup 方式，使用方式与 tensorflow 原生接口一致
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
