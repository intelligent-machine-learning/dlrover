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

import tensorflow.compat.v1 as tf

from dlrover.trainer.util.log_util import default_logger as logger

tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.INFO)


class MyEstimator(tf.estimator.Estimator):
    """MyEstimator"""

    def __init__(self, model_dir, config=None, params=None):

        logger.info("config is %s", config)
        logger.info("model_dir is %s", config)
        run_config = config

        super(MyEstimator, self).__init__(
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
        optimizer = tf.train.AdamOptimizer()
        x = features["x"]
        w = tf.Variable(0.1, name="x")
        b = tf.Variable(0.1, name="b")
        prediction = w * x + b
        print("Mode = ", mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=prediction)

        loss = tf.losses.mean_squared_error(labels, prediction)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_or_create_global_step()
        )
        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {
                "mse": tf.metrics.mean_squared_error(labels, prediction)
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
