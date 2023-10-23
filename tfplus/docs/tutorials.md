# End to End Training Tutorials
<div id="top" align="center">

   | [English](./tutorials.md) | [中文](./tutorials_ZH.md) |

</div>

In this tutorial, we will demonstrate how to perform end-to-end model training using TFPlus's Kv Variable and custom optimizer.
## Case 1: Recommending movies: ranking
>This case refers to the embedding_variable_tutorial in the tensorflow/recommenders-addons repository. For details, see: https://github.com/tensorflow/recommenders-addons/blob/master/docs/tutorials/embedding_variable_tutorial.ipynb

To download the dataset, you need to install ﻿tensorflow_datasets separately:
```shell
pip install tensorflow_datasets
```

In this case, we use the MovieLens 100K dataset to implement a neural collaborative filtering(NeuralCF) model to train a scoring model.
>Note: For now, all examples can only run under Graph Mode.

First, we read the MovieLens dataset, preprocess the data, and create a tensorflow dataset. In order to facilitate processing, we convert the data type of `movie_id` and `user_id` into `int64`.

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
Next, we construct a NeuralCF model. In the model definition, we use **tfplus.kv_variable.python.ops.variable_scope.get_kv_variable** to build the embedding layer and tfplus.**kv_variable.python.ops.embedding_ops.embedding_lookup** to achieve the embedding feature lookup.
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
Finally, we instantiate this model, use the custom Adam optimizer **tfplus.kv_variable.python.training.adam.AdamOptimizer**, and update the model's gradient.
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
For complete code, see: [example/NCFModel/train.py](../example/NCFModel/train.py)

## Case2: DCN for Advertising CTR Prediction
see: [example/dcn](../example/dcn/README.md)