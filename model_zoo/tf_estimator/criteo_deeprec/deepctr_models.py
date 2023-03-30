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

import json
import os

import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
from layers import DNN, FM, Discretization, Hashing, Normalizer

from dlrover.trainer.util.log_util import default_logger as logger

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.INFO)


CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = "label"
TRAIN_DATA_COLUMNS = [LABEL_COLUMN] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

DEEP_EMBEDDING_DIM = 8
HIDDEN_UNITS = [16, 4]
ACTIVATION = "relu"
LEARNING_RATE = 0.0001
EPSILON = 1e-05


class NumericStats(object):
    def __init__(self, avg, stddev, bucketize_values) -> None:
        self.avg = avg
        self.stddev = stddev
        self.bucketize_values = bucketize_values


_NUMERIC_FEATURE_STATS = {
    "I1": NumericStats(3.6, 14.5, [0.0, 1.0, 2.0, 3.0, 5.0, 9.0]),
    "I2": NumericStats(
        86.7, 328.6, [-1.0, 0.0, 1.0, 2.0, 7.0, 19.0, 48.0, 163.0]
    ),
    "I3": NumericStats(
        26.1, 493.4, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0, 30.0]
    ),
    "I4": NumericStats(6.6, 8.0, []),
    "I5": NumericStats(
        17917.2,
        67045.7,
        [
            14.0,
            691.0,
            1051.0,
            1402.0,
            2700.0,
            5325.0,
            7670.0,
            12915.0,
            34205.0,
        ],
    ),
    "I6": NumericStats(
        111.7, 350.1, [1.0, 5.0, 11.0, 19.0, 31.0, 49.0, 77.0, 132.0, 265.0]
    ),
    "I7": NumericStats(16.6, 69.2, [0.0, 1.0, 2.0, 4.0, 6.0, 9.0, 16.0, 34.0]),
    "I8": NumericStats(
        13.3, 17.5, [0.0, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 25.0, 36.0]
    ),
    "I9": NumericStats(
        106.6, 210, [3.0, 8.0, 15.0, 26.0, 41.0, 61.0, 92.0, 147.0, 268.0]
    ),
    "I10": NumericStats(0.6, 0.7, [0.0, 1.0]),
    "I11": NumericStats(2.9, 5.4, [0.0, 1.0, 2.0, 4.0, 7.0]),
    "I12": NumericStats(1.0, 6.2, [0.0, 1.0]),
    "I13": NumericStats(6.5, 15.2, [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 16.0]),
}


_CATEGORY_FEATURE_STATS = {
    "C1": 1036,
    "C2": 530,
    "C3": 169550,
    "C4": 71524,
    "C5": 241,
    "C6": 15,
    "C7": 10025,
    "C8": 458,
    "C9": 3,
    "C10": 22960,
    "C11": 4469,
    "C12": 144780,
    "C13": 3034,
    "C14": 26,
    "C15": 7577,
    "C16": 113860,
    "C17": 10,
    "C18": 3440,
    "C19": 1678,
    "C20": 3,
    "C21": 130892,
    "C22": 11,
    "C23": 14,
    "C24": 27189,
    "C25": 65,
    "C26": 20188,
}


class WideAndDeep(tf.estimator.Estimator):
    """Wide and Deep"""

    def __init__(self, model_dir, config=None, params=None):

        logger.info("config is %s", config)
        logger.info("model_dir is %s", config)
        run_config = config

        super(WideAndDeep, self).__init__(
            self.model_fn,
            model_dir=model_dir,
            config=run_config,
            params=params,
        )

    def model_fn(self, features, labels, mode, params):
        """
        featurs: type dict, key is the feature name and value is tensor.
                 In this case. it is like {"x": Tensor}.
        labels: type tensor, corresponding to the colum which `is_label` equals True. # noqa: E501
                In this case, it is like Tensor.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            K.set_learning_phase(1)
        else:
            K.set_learning_phase(0)

        features = reshape_features(features)
        dense_tensor, categroy_tensors = transform_feature(features)

        linear_logits = lookup_embedding_func(
            categroy_tensors,
            embedding_dim=1,
        )

        # deep part
        deep_embeddings = lookup_embedding_func(
            categroy_tensors,
            embedding_dim=DEEP_EMBEDDING_DIM,
        )

        # Deep Part
        dnn_input = tf.concat(deep_embeddings, axis=1)
        if dense_tensor is not None:
            dnn_input = tf.keras.layers.Concatenate()(
                [dense_tensor, dnn_input]
            )
            linear_logits.append(
                tf.keras.layers.Dense(1, activation=None, use_bias=False)(
                    dense_tensor
                )
            )

        # Linear Part
        if len(linear_logits) > 1:
            linear_logit = tf.keras.layers.Concatenate()(linear_logits)
        else:
            linear_logit = linear_logits[0]

        hidden_units = HIDDEN_UNITS
        activation = ACTIVATION
        dnn_output = DNN(hidden_units, activation)(dnn_input)

        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
            dnn_output
        )

        # Output Part
        concat_input = tf.concat([linear_logit, dnn_logit], 1)

        logits = tf.reduce_sum(concat_input, 1, keepdims=True)
        probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))
        return build_estimator(logits, probs, labels, mode)


class DeepFM(tf.estimator.Estimator):
    """MyEstimator"""

    def __init__(self, model_dir, config=None, params=None):

        logger.info("config is %s", config)
        logger.info("model_dir is %s", config)
        run_config = config

        super(DeepFM, self).__init__(
            self.model_fn,
            model_dir=model_dir,
            config=run_config,
            params=params,
        )

    def model_fn(self, features, labels, mode, params):
        """
        featurs: type dict, key is the feature name and value is tensor.
                 In this case. it is like {"x": Tensor}.
        labels: type tensor, corresponding to the colum which `is_label` equals True. # noqa: E501
                In this case, it is like Tensor.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            K.set_learning_phase(1)
        else:
            K.set_learning_phase(0)
        features = reshape_features(features)
        dense_tensor, categroy_tensors = transform_feature(features)

        linear_logits = lookup_embedding_func(
            categroy_tensors,
            embedding_dim=1,
        )

        # deep part
        deep_embeddings = lookup_embedding_func(
            categroy_tensors,
            embedding_dim=DEEP_EMBEDDING_DIM,
        )

        # Deep Part
        dnn_input = tf.concat(deep_embeddings, axis=1)
        if dense_tensor is not None:
            dnn_input = tf.keras.layers.Concatenate()(
                [dense_tensor, dnn_input]
            )
            linear_logits.append(
                tf.keras.layers.Dense(1, activation=None, use_bias=False)(
                    dense_tensor
                )
            )

        # Linear Part
        if len(linear_logits) > 1:
            linear_logit = tf.keras.layers.Concatenate()(linear_logits)
        else:
            linear_logit = linear_logits[0]

        hidden_units = HIDDEN_UNITS
        activation = ACTIVATION
        dnn_output = DNN(hidden_units, activation)(dnn_input)
        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
            dnn_output
        )

        if len(deep_embeddings) > 1:
            FM_output = FM(embedding_dim=DEEP_EMBEDDING_DIM)(deep_embeddings)
            # Output Part
            concat_input = tf.concat([linear_logit, dnn_logit, FM_output], 1)
        else:
            concat_input = tf.concat([linear_logit, dnn_logit], 1)

        logits = tf.reduce_sum(concat_input, 1, keepdims=True)
        probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))
        return build_estimator(logits, probs, labels, mode)


class xDeepFM(tf.estimator.Estimator):
    """Wide and Deep"""

    def __init__(self, model_dir, config=None, params=None):

        logger.info("config is %s", config)
        logger.info("model_dir is %s", config)
        run_config = config

        super(xDeepFM, self).__init__(
            self.model_fn,
            model_dir=model_dir,
            config=run_config,
            params=params,
        )

    def model_fn(self, features, labels, mode, params):
        """
        featurs: type dict, key is the feature name and value is tensor.
                 In this case. it is like {"x": Tensor}.
        labels: type tensor, corresponding to the colum which `is_label` equals True. # noqa: E501
                In this case, it is like Tensor.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            K.set_learning_phase(1)
        else:
            K.set_learning_phase(0)

        features = reshape_features(features)
        dense_tensor, categroy_tensors = transform_feature(features)

        linear_logits = lookup_embedding_func(
            categroy_tensors,
            embedding_dim=1,
        )

        # deep part
        deep_embeddings = lookup_embedding_func(
            categroy_tensors,
            embedding_dim=DEEP_EMBEDDING_DIM,
        )

        dnn_input = tf.concat(deep_embeddings, axis=1)
        if dense_tensor is not None:
            dnn_input = tf.keras.layers.Concatenate()(
                [dense_tensor, dnn_input]
            )
            linear_logits.append(
                tf.keras.layers.Dense(1, activation=None, use_bias=False)(
                    dense_tensor
                )
            )

        # Linear Part
        if len(linear_logits) > 1:
            linear_logit = tf.keras.layers.Concatenate()(linear_logits)
        else:
            linear_logit = linear_logits[0]

        hidden_units = HIDDEN_UNITS
        activation = ACTIVATION
        dnn_output = DNN(hidden_units, activation)(dnn_input)

        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
            dnn_output
        )

        if len(deep_embeddings) > 1:
            field_size = len(deep_embeddings)
            embeddings = tf.concat(
                deep_embeddings, 1
            )  # shape = (None, field_size, 8)
            embeddings = tf.reshape(
                embeddings, shape=(-1, field_size, DEEP_EMBEDDING_DIM)
            )
            from deepctr.layers.interaction import CIN

            exFM_out = CIN(
                layer_size=(
                    128,
                    128,
                ),
                activation="relu",
                split_half=True,
            )(embeddings)
            exFM_logit = tf.keras.layers.Dense(
                1,
                activation=None,
            )(exFM_out)
            # Output Part
            concat_input = tf.concat([linear_logit, dnn_logit, exFM_logit], 1)
        else:
            concat_input = tf.concat([linear_logit, dnn_output], 1)

        logits = tf.reduce_sum(concat_input, 1, keepdims=True)
        probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))

        return build_estimator(logits, probs, labels, mode)


def reshape_features(features):
    input_tensors = {}
    for name, tensor in features.items():
        input_tensors[name] = tf.reshape(tensor, (-1, 1))
    return input_tensors


def build_estimator(logits, probs, labels, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "probs": probs,
            "logits": logits,
        }
        export_outputs = {
            "prediction": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs,
        )

    labels = tf.cast(labels, tf.float32)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    # Compute evaluation metrics.
    auc = tf.metrics.auc(labels, probs, name="auc_op")

    metrics = {"auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE, epsilon=EPSILON
    )
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def transform_feature(inputs):
    """According to the FeatureConfig object and feature groups to
    transform inputs to dense tensors.

    Args:
        inputs: A dict contains Keras inputs where the key is the
            feature name and the value is the Keras input.

    Returns:
        standarized_tensor: A float tensor
        category_tensors: A dict where the key is the feature
            and the value is tensor.
    """

    standardized_outputs = []
    for feature in _NUMERIC_FEATURE_STATS.keys():
        standardized_result = Normalizer(
            subtractor=_NUMERIC_FEATURE_STATS[feature].avg,
            divisor=_NUMERIC_FEATURE_STATS[feature].stddev,
        )(inputs[feature])
        standardized_outputs.append(standardized_result)

    numerical_tensor = (
        tf.concat(standardized_outputs, -1) if standardized_outputs else None
    )

    category_tensors = {}
    for feature in _NUMERIC_FEATURE_STATS.keys():
        discretize_layer = Discretization(
            bins=_NUMERIC_FEATURE_STATS[feature].bucketize_values
        )
        category_tensors[feature] = discretize_layer(inputs[feature])

    for feature in CATEGORICAL_COLUMNS:
        num_bins = _CATEGORY_FEATURE_STATS[feature]
        hash_layer = Hashing(num_bins=num_bins)
        hash_output = hash_layer(inputs[feature])
        category_tensors[feature] = hash_output

    return numerical_tensor, category_tensors


def lookup_embedding_func(input_tensors, embedding_dim):
    """
    Args:
        input_tensors: dict, the key is a string and the value
            is a tensor outputed by the transform function
        deep_embedding_dim: The output dimension of embedding layer for
            deep parts.
    """
    TF_CONFIG = json.loads(os.environ["TF_CONFIG"])
    ps_cluster = TF_CONFIG.get("cluster").get("ps")
    ps_num = len(ps_cluster)

    embeddings = []
    for name, tensor in input_tensors.items():
        var_name = name + "_embedding_" + str(embedding_dim)

        var = tf.get_embedding_variable(
            var_name,
            embedding_dim=embedding_dim,
            key_dtype=tf.int64,
            initializer=tf.ones_initializer(tf.float32),
            partitioner=tf.fixed_size_partitioner(num_shards=ps_num),
            ev_option=tf.EmbeddingVariableOption(),
        )

        embedding = tf.nn.embedding_lookup(var, tensor)
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        embeddings.append(embedding_sum)
    return embeddings
