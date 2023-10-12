# TFPlus 端到端训练教程
<div id="top" align="center">

   | [English](./tutorials.md) | [中文](./tutorials_ZH.md) |

</div>

这个教程中，我们会展示如何使用 TFPlus 的 Kv Variable 和定制优化器实现端到端的模型训练。
## Case 1: Recommending movies: ranking
> 这个 case 参考了 tensorflow/recommenders-addons 仓库的 embedding_variable_tutorial，详见：https://github.com/tensorflow/recommenders-addons/blob/master/docs/tutorials/embedding_variable_tutorial.ipynb

在这个 case 中，我们使用 MovieLens 100K dataset 实现神经协同过滤（NeuralCF）模型来训练一个评分模型。

为了下载数据集，你需要额外安装tensorflow_datasets:
```shell
pip install tensorflow_datasets
```
> 注：所有的例子目前都只能在 Graph Mode 下运行

首先我们读取 MovieLens 数据集，进行数据预处理, 创建 tensorflow dataset，为了方便处理，我们将 `movie_id` 和 `user_id` 的数据类型转换为 `int64`。

```python
ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(
    lambda x: {
        "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
        "user_id": tf.strings.to_number(x["user_id"], tf.int64),
        "user_rating": x["user_rating"]
    })

tf.random.set_seed(2021)
shuffled = ratings.shuffle(100_000, seed=2021, reshuffle_each_iteration=False)
dataset_train = shuffled.take(100_000).batch(256)

iterator = tf.compat.v1.data.make_one_shot_iterator(dataset_train)
dataset_train = iterator.get_next()
```
然后我们构建一个 NeuralCF model，在模型定义中，使用 **tfplus.kv_variable.python.ops.variable_scope.get_kv_variable** 来构建 embedding layer，使用 **tfplus.kv_variable.python.ops.embedding_ops.embedding_lookup** 来实现 embedding 特征的查找

```python
from tfplus.kv_variable.python.ops.variable_scope import get_kv_variable
from tfplus.kv_variable.python.ops.embedding_ops import embedding_lookup

class NCFModel(tf.keras.Model):
  def __init__(self):
    super(NCFModel, self).__init__()
    self.embedding_size = 32
    self.d0 = Dense(
        256,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d1 = Dense(
        64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d2 = Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.user_embeddings = get_kv_variable(
        name="user_dynamic_embeddings",
        key_dtype=tf.int64,
        embedding_dim=self.embedding_size,
        initializer=tf.compat.v1.keras.initializers.RandomNormal(-1, 1))
    self.movie_embeddings = get_kv_variable(
        name="moive_dynamic_embeddings",
        embedding_dim=self.embedding_size,
        key_dtype=tf.int64,
        initializer=tf.compat.v1.keras.initializers.RandomNormal(-1, 1))
    self.loss = tf.keras.losses.MeanSquaredError()

  def call(self, batch):
    movie_id = batch["movie_id"]
    user_id = batch["user_id"]
    rating = batch["user_rating"]

    user_id_val, user_id_idx = tf.unique(user_id)
    user_id_weights = embedding_lookup(params=self.user_embeddings,
                                             ids=user_id_val,
                                             name="user-id-weights")
    user_id_weights = tf.gather(user_id_weights, user_id_idx)
    movie_id_val, movie_id_idx = tf.unique(movie_id)
    movie_id_weights = embedding_lookup(params=self.movie_embeddings,
                                              ids=movie_id_val,
                                              name="movie-id-weights")
    movie_id_weights = tf.gather(movie_id_weights, movie_id_idx)
    embeddings = tf.concat([tf.reshape(user_id_weights, [-1, self.embedding_size]), tf.reshape(movie_id_weights, [-1, self.embedding_size])], axis=1)
    dnn = self.d0(embeddings)
    dnn = self.d1(dnn)
    dnn = self.d2(dnn)
    out = tf.reshape(dnn, shape=[-1])
    loss = self.loss(rating, out)
    return loss
```
最后我们实例化这个模型，使用定制的 Adam 优化器 **tfplus.kv_variable.python.training.adam.AdamOptimizer** 并进行模型的梯度更新
```python
from tfplus.kv_variable.python.training.adam import AdamOptimizer
model = NCFModel()
loss = model(dataset_train)
optimizer = AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)
global_init = tf.compat.v1.global_variables_initializer()

epoch = 20
with tf.compat.v1.Session() as sess:
  sess.run(global_init)
  for i in range(epoch):
    loss_t, _ = sess.run([loss, train_op])
    print("epoch:", i, "loss:", loss_t)
```
完整的代码见：[example/NCFModel/train.py](../example/NCFModel/train.py)

## Case2: DCN for Advertising CTR Prediction
详见: [example/dcn](../example/dcn/README.md)