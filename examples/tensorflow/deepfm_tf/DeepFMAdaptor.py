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
from deepctr.estimator.models import DeepFMEstimator

from dlrover.trainer.util.log_util import default_logger as logger

tf.disable_v2_behavior()

tf.logging.set_verbosity(tf.logging.INFO)


class DeepFMAdaptor(tf.estimator.Estimator):
    """Adaptor"""

    def __init__(self, model_dir, config=None, params=None):

        logger.info("config is %s", config)
        self.run_config = config
        self.estimator = None

        super(DeepFMAdaptor, self).__init__(
            self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
        )

    def model_fn(self, features, labels, mode, params):
        sparse_features = params["sparse_features"]
        dense_features = params["dense_features"]
        dnn_feature_columns = []
        linear_feature_columns = []
        for i, feat in enumerate(sparse_features):
            dnn_feature_columns.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_hash_bucket(
                        feat, 1000
                    ),
                    4,
                )
            )
            linear_feature_columns.append(
                tf.feature_column.categorical_column_with_hash_bucket(
                    feat, 1000
                )
            )
        for feat in dense_features:
            dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
            linear_feature_columns.append(
                tf.feature_column.numeric_column(feat)
            )

        if self.estimator is None:
            self.estimator = DeepFMEstimator(
                linear_feature_columns,
                dnn_feature_columns,
                task=params["task"],
            )
        return self.estimator._model_fn(
            features, labels, mode, self.run_config
        )
