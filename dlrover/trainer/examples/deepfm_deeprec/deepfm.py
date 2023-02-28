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
from tensorflow.python.ops import partitioned_variables

from dlrover.trainer.util.log_util import default_logger as logger

tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.INFO)


CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["label"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
dnn_hidden_units = [1024, 256, 32]
final_hidden_units = [128, 64]

TF_CONFIG = json.loads(os.environ["TF_CONFIG"])
ps_cluster = TF_CONFIG.get("cluster").get("ps")
ps_num = len(ps_cluster)

input_layer_partitioner = partitioned_variables.fixed_size_partitioner(
    ps_num, 0
)


def build_feature_columns():
    wide_column = []
    deep_column = []
    fm_column = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            evict_opt = None
            filter_option = None
            ev_opt = tf.EmbeddingVariableOption(
                evict_option=evict_opt, filter_option=filter_option
            )
            categorical_column = (
                tf.feature_column.categorical_column_with_embedding(
                    column_name, dtype=tf.string, ev_option=ev_opt
                )
            )
            embedding_column = tf.feature_column.embedding_column(
                categorical_column, dimension=16, combiner="mean"
            )
            wide_column.append(embedding_column)
            deep_column.append(embedding_column)
            fm_column.append(embedding_column)
        else:
            column = tf.feature_column.numeric_column(column_name, shape=(1,))
            wide_column.append(column)
            deep_column.append(column)

    return wide_column, fm_column, deep_column


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
        wide_column, fm_column, deep_column = build_feature_columns()

        # input_layer_partitioner = None
        with tf.variable_scope(
            "input_layer",
            partitioner=input_layer_partitioner,
            reuse=tf.AUTO_REUSE,
        ):
            fm_cols = {}
            dnn_input = tf.feature_column.input_layer(features, deep_column)
            wide_input = tf.feature_column.input_layer(
                features, wide_column, cols_to_output_tensors=fm_cols
            )

            fm_input = tf.stack([fm_cols[cols] for cols in fm_column], 1)

        layer_name = ""
        with tf.variable_scope("dnn"):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with tf.variable_scope(
                    layer_name + "_%d" % layer_id, reuse=tf.AUTO_REUSE
                ) as dnn_layer_scope:
                    dnn_input = tf.layers.dense(
                        dnn_input,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=dnn_layer_scope,
                    )
        dnn_output = dnn_input
        with tf.variable_scope("linear", reuse=tf.AUTO_REUSE):
            linear_output = tf.reduce_sum(wide_input, axis=1, keepdims=True)

        # FM second order part
        with tf.variable_scope("fm", reuse=tf.AUTO_REUSE):
            sum_square = tf.square(tf.reduce_sum(fm_input, axis=1))
            square_sum = tf.reduce_sum(tf.square(fm_input), axis=1)
            fm_output = 0.5 * tf.subtract(sum_square, square_sum)

        # Final dnn layer
        all_input = tf.concat([dnn_output, linear_output, fm_output], 1)
        dnn_input = all_input
        with tf.variable_scope("final_dnn"):
            for layer_id, num_hidden_units in enumerate(final_hidden_units):
                with tf.variable_scope(
                    layer_name + "_%d" % layer_id,
                    partitioner=input_layer_partitioner,
                    reuse=tf.AUTO_REUSE,
                ) as dnn_layer_scope:
                    dnn_input = tf.layers.dense(
                        dnn_input,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=dnn_layer_scope,
                    )

        logits = tf.layers.dense(dnn_input, units=1)
        probability = tf.math.sigmoid(logits)

        loss_func = tf.losses.mean_squared_error
        prediction = tf.squeeze(probability)
        loss = tf.math.reduce_mean(loss_func(labels, prediction))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=prediction)

        loss = tf.losses.mean_squared_error(labels, prediction)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_or_create_global_step()
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            metrics = {
                "auc": tf.metrics.auc(
                    labels=labels, predictions=probability, num_thresholds=1000
                )
            }

            return tf.estimator.EstimatorSpec(
                mode,
                predictions=prediction,
                eval_metric_ops=metrics,
                loss=loss,
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode, predictions=prediction, loss=loss, train_op=train_op
            )

        raise ValueError("Not a valid mode: {}".format(mode))
