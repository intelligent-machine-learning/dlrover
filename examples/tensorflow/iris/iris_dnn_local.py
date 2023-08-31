# Copyright 2021 The DLRover Authors. All rights reserved.
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

import csv
import os

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

CATEGORY_CODE = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
DATASET_DIR = "examples/tensorflow/iris/iris.data"


def read_csv(file_path):
    rows = []
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)
    return rows


def model_fn(features, labels, mode, params):
    net = tf.compat.v1.feature_column.input_layer(
        features, params["feature_columns"]
    )

    for units in params["hidden_units"]:
        net = tf.compat.v1.layers.dense(
            net, units=units, activation=tf.nn.relu
        )
    logits = tf.compat.v1.layers.dense(
        net, params["n_classes"], activation=None
    )

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": predicted_classes[:, tf.newaxis],
            "probs": tf.nn.softmax(logits),
            "logits": logits,
        }
        export_outputs = {
            "prediction": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs
        )

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss, global_step=tf.compat.v1.train.get_global_step()
        )
        logging_hook = tf.compat.v1.train.LoggingTensorHook(
            {"loss": loss}, every_n_iter=10
        )
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op, training_hooks=[logging_hook]
        )

    accuracy = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name="acc"
    )
    metrics = {"accuracy": accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )


def train_generator():
    rows = read_csv(DATASET_DIR)
    for i in range(100):
        for row in rows:
            if len(row) > 0:
                label = CATEGORY_CODE[row[-1]]
                yield row[0:-1], [label]


def eval_generator():
    rows = read_csv(DATASET_DIR)
    for row in rows:
        if len(row) > 0:
            label = CATEGORY_CODE[row[-1]]
            yield row[0:-1], [label]


def input_fn(sample_generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        sample_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(4, 1),
    )
    dataset = dataset.shuffle(100).batch(batch_size)
    data_it = tf.compat.v1.data.make_one_shot_iterator(dataset)
    feature_values, label_values = data_it.get_next()
    features = {"x": feature_values}
    return features, label_values


if __name__ == "__main__":
    model_dir = "/data/ckpts/"
    batch_size = 64
    feature_columns = [
        tf.feature_column.numeric_column(key="x", shape=(4,), dtype=tf.float32)
    ]
    os.makedirs(model_dir, exist_ok=True)

    config = tf.estimator.RunConfig(
        model_dir=model_dir, save_checkpoints_steps=300, keep_checkpoint_max=5
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params={
            "hidden_units": [8, 4],
            "n_classes": 3,
            "feature_columns": feature_columns,
        },
    )

    # Create a data shard service which can split the dataset
    # into shards.
    rows = read_csv(DATASET_DIR)

    def train_input_fn():
        return input_fn(lambda: train_generator(), batch_size)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_generator, batch_size)
    )
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
