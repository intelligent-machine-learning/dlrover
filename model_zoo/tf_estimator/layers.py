# Copyright 2023 The DLRover Authors. All rights reserved.
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

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops


class Normalizer(tf.keras.layers.Layer):
    """Normalize the numeric tensors by (x-subtractor)/divisor

    Example :
    ```python
        layer = Normalizer(subtractor=1.0, divisor=2.0)
        inp = np.asarray([[3.0], [5.0], [7.0]])
        layer(inp)
        [[1.0], [2.0], [3.0]]
    ```

    Arguments:
        subtractor: A float value.
        divisor: A float value.

    Input shape: A numeric `Tensor`, `SparseTensor` or `RaggedTensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An float64 tensor of shape `[batch_size, d1, ..., dm]`

    """

    def __init__(self, subtractor, divisor, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.subtractor = subtractor
        self.divisor = divisor

    def build(self, input_shape):
        if self.divisor == 0:
            raise ValueError("The divisor cannot be 0")

    def get_config(self):
        config = {"subtractor": self.subtractor, "divisor": self.divisor}
        base_config = super(Normalizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            normalized_tensor = tf.ragged.map_flat_values(
                self._normalize_fn, inputs
            )
        elif isinstance(inputs, tf.SparseTensor):
            normalize_values = self._normalize_fn(inputs.values)
            normalized_tensor = tf.SparseTensor(
                indices=inputs.indices,
                values=normalize_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            normalized_tensor = self._normalize_fn(inputs)

        return normalized_tensor

    def _normalize_fn(self, x):
        x = tf.cast(x, tf.float32)
        subtractor = tf.cast(self.subtractor, tf.float32)
        divisor = tf.cast(self.divisor, tf.float32)
        return (x - subtractor) / divisor


class Discretization(tf.keras.layers.Layer):
    """Buckets data into discrete ranges.

    TensorFlow 2.2 has developed `tf.keras.layers.preprocessing.Discretization`
    but not released it yet. So the layer is a simple temporary version
    `tensorflow.python.keras.layers.preprocessing.discretization.Discretization`

    Input shape:
        Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

    Output shape:
        The same as the input shape with tf.int64.

    Attributes:
        bins: Optional boundary specification. Bins include the left boundary
            and exclude the right boundary, so `bins=[0., 1., 2.]` generates
            bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.
    """

    def __init__(self, bins, **kwargs):
        super(Discretization, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.bins = bins

    def num_bins(self):
        """The bins is a list with boundaries, so the number of bins is
        len(bins) + 1.
        """
        return len(self.bins) + 1

    def get_config(self):
        config = {
            "bins": self.bins,
        }
        base_config = super(Discretization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            integer_buckets = tf.ragged.map_flat_values(
                math_ops._bucketize, inputs, boundaries=self.bins
            )
            integer_buckets = tf.identity(integer_buckets)
        elif isinstance(inputs, tf.SparseTensor):
            integer_bucket_values = math_ops._bucketize(
                inputs.values, boundaries=self.bins
            )
            integer_buckets = tf.SparseTensor(
                indices=inputs.indices,
                values=integer_bucket_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            integer_buckets = math_ops._bucketize(inputs, boundaries=self.bins)

        return tf.cast(integer_buckets, tf.int64)


class DNN(tf.keras.layers.Layer):
    """DNN layer

    Args:
        hidden_units: A list with integers
        activation: activation function name, like "relu"
    inputs: A Tensor
    """

    def __init__(self, hidden_units, activation=None, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.dense_layers = []
        if not hidden_units:
            raise ValueError("The hidden units cannot be empty")
        for hidden_unit in hidden_units:
            self.dense_layers.append(
                tf.keras.layers.Dense(hidden_unit, activation=activation)
            )

    def call(self, inputs):
        output = inputs
        for layer in self.dense_layers:
            output = layer(output)
        return output


class FM(tf.keras.layers.Layer):
    """FM

    Inputs: A list of embedding output tensors with the
        same shape.
    """

    def __init__(self, embedding_dim=8, **kwargs):
        self._embedding_dim = embedding_dim
        super(FM, self).__init__(**kwargs)

    def call(self, inputs):
        from tensorflow.keras.layers import Subtract

        group_num = len(inputs)
        # FM Part
        embeddings = tf.concat(inputs, 1)  # shape = (None, group_num , 8)
        embeddings = tf.reshape(
            embeddings, shape=(-1, group_num, self._embedding_dim)
        )  # shape = (None, group_num, 8)
        emb_sum = K.sum(embeddings, axis=1)  # shape = (None, 8)
        emb_sum_square = K.square(emb_sum)  # shape = (None, 8)
        emb_square = K.square(embeddings)  # shape = (None, group_num, 8)
        emb_square_sum = K.sum(emb_square, axis=1)  # shape = (None, 8)
        FM = 0.5 * Subtract()(
            [emb_sum_square, emb_square_sum]
        )  # shape = (None, 8)
        return FM
