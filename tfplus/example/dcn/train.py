# Copyright 2023 The TFPlus Authors. All rights reserved.
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

"""Train DCN model on Criteo dataset."""
import argparse
import collections
import math
import os
import sys
import time

import tensorflow as tf

from tfplus.kv_variable.python.ops import kv_variable_ops
from tfplus.kv_variable.python.ops.embedding_ops import embedding_lookup
from tfplus.kv_variable.python.ops.variable_scope import get_kv_variable
from tfplus.kv_variable.python.training.adagrad import AdagradOptimizer
from tfplus.kv_variable.python.training.adam import AdamOptimizer
from tfplus.kv_variable.python.training.group_adam import GroupAdamOptimizer
from tfplus.kv_variable.python.training.sparse_group_ftrl import (
    SparseGroupFtrlOptimizer,
)

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
print("Using TensorFlow version %s" % tf.__version__)
tf.compat.v1.disable_v2_behavior()

# Definition of some constants
CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
    "C1": 2500,
    "C2": 2000,
    "C3": 300000,
    "C4": 250000,
    "C5": 1000,
    "C6": 100,
    "C7": 20000,
    "C8": 4000,
    "C9": 20,
    "C10": 100000,
    "C11": 10000,
    "C12": 250000,
    "C13": 40000,
    "C14": 100,
    "C15": 100,
    "C16": 200000,
    "C17": 50,
    "C18": 10000,
    "C19": 4000,
    "C20": 20,
    "C21": 250000,
    "C22": 100,
    "C23": 100,
    "C24": 250000,
    "C25": 400,
    "C26": 100000,
}

EMBEDDING_DIMENSIONS = {
    "C1": 64,
    "C2": 64,
    "C3": 128,
    "C4": 128,
    "C5": 64,
    "C6": 64,
    "C7": 64,
    "C8": 64,
    "C9": 64,
    "C10": 128,
    "C11": 64,
    "C12": 128,
    "C13": 64,
    "C14": 64,
    "C15": 64,
    "C16": 128,
    "C17": 64,
    "C18": 64,
    "C19": 64,
    "C20": 64,
    "C21": 128,
    "C22": 64,
    "C23": 64,
    "C24": 128,
    "C25": 64,
    "C26": 128,
}


class DCN:
  """DCN model."""
  def __init__(
      self,
      emb_stacking_columns=None,
      dnn_hidden_units=None,
      optimizer_type="adam",
      cross_learning_rate=0.001,
      deep_learning_rate=0.001,
      inputs=None,
      bf16=False,
      stock_tf=None,
      input_layer_partitioner=None,
      dense_layer_partitioner=None,
  ):
    if dnn_hidden_units is None:
      dnn_hidden_units = [1024, 512, 256]
    if not inputs:
      raise ValueError("Dataset is not defined.")
    self._feature = inputs[0]
    self._label = inputs[1]
    self._emb_stacking_columns = emb_stacking_columns
    if not emb_stacking_columns:
      raise ValueError("Embedding and stacking column is not defined.")

    self.tf = stock_tf
    self.bf16 = False if self.tf else bf16
    self.is_training = True

    self._dnn_hidden_units = dnn_hidden_units
    self._deep_learning_rate = deep_learning_rate
    self._cross_learning_rate = cross_learning_rate
    self._optimizer_type = optimizer_type
    self._input_layer_partitioner = input_layer_partitioner
    self._dense_layer_partitioner = dense_layer_partitioner
    (
        self._categorical_columns,
        self._numerical_columns,
    ) = self._emb_stacking_columns

    self._create_model()
    with tf.compat.v1.name_scope("head"):
      self._create_loss()
      self._create_optimizer()
      self._create_metrics()

  # used to add summary in tensorboard
  @staticmethod
  def _add_layer_summary(value, tag):
    tf.compat.v1.summary.scalar("%s/fraction_of_zero_values" % tag,
                                tf.nn.zero_fraction(value))
    tf.compat.v1.summary.histogram("%s/activation" % tag, value)

  def _dnn(self, dnn_input, dnn_hidden_units=None, layer_name=""):
    """dnn part"""
    for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
      with tf.compat.v1.variable_scope(
          layer_name + "_%d" % layer_id,
          partitioner=self._dense_layer_partitioner,
          reuse=tf.compat.v1.AUTO_REUSE,
      ) as dnn_layer_scope:
        dnn_input = tf.compat.v1.layers.dense(
            dnn_input,
            units=num_hidden_units,
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
            name=dnn_layer_scope,
        )

        self._add_layer_summary(dnn_input, dnn_layer_scope.name)
    return dnn_input

  def _cross_net(self, cross_input, layer_num=2, layer_name=""):
    """cross net part"""
    # add diag_scale
    x = x0 = cross_input
    last_dim = cross_input.shape[-1]

    for layer_id in range(layer_num):
      with tf.compat.v1.variable_scope(
          layer_name + "_%d" % layer_id,
          partitioner=self._dense_layer_partitioner,
          reuse=tf.compat.v1.AUTO_REUSE,
      ) as cross_layer_scope:
        w = tf.compat.v1.get_variable(
            name=layer_name + "_w",
            dtype=cross_input.dtype,
            shape=(last_dim),
        )
        b = tf.compat.v1.get_variable(
            name=layer_name + "_b",
            dtype=cross_input.dtype,
            shape=(last_dim),
        )
        xw = tf.reduce_sum(x * w, axis=1, keepdims=True)
        x = tf.math.add(tf.math.add(x0 * xw, b), x)

        self._add_layer_summary(cross_input, cross_layer_scope.name)
    return x


  def _create_model(self):
    '''create model'''
    # Dnn part
    with tf.compat.v1.variable_scope("dnn"):
      # input layer
      with tf.compat.v1.variable_scope(
          "input_from_feature_columns",
          partitioner=self._input_layer_partitioner,
          reuse=tf.compat.v1.AUTO_REUSE,
      ):
        # Process categorical columns
        categorical_net_indexed = []
        for i in range(1, 27):
          col_name = "C" + str(i)
          indexed_column = tf.strings.to_hash_bucket_fast(
              self._feature[col_name], HASH_BUCKET_SIZES[col_name])
          # Embed categorical features
          dnn_embedding = get_kv_variable(
              name="embedding_weight_{}".format(i),
              key_dtype=tf.int64,
              embedding_dim=EMBEDDING_DIMENSIONS[col_name],
              initializer=tf.compat.v1.keras.initializers.RandomNormal(-1, 1),
          )
          embedded_column = embedding_lookup(
              params=dnn_embedding,
              ids=indexed_column,
              name="embedding_column_{}".format(i),
          )
          categorical_net_indexed.append(embedded_column)

        categorical_net = tf.concat(values=categorical_net_indexed, axis=-1)
        numerical_net = tf.compat.v1.feature_column.input_layer(
            features={
                numerical_column: self._feature[numerical_column]
                for numerical_column in CONTINUOUS_COLUMNS
            },
            feature_columns=[
                self._numerical_columns[numerical_column]
                for numerical_column in CONTINUOUS_COLUMNS
            ],
        )

        # Concatenate
        net = tf.concat([categorical_net, numerical_net], axis=-1)
        self._add_layer_summary(net, "input_from_feature_columns")

      # hidden layers
      dnn_scope = tf.compat.v1.variable_scope(
          "dnn_layers",
          partitioner=self._dense_layer_partitioner,
          reuse=tf.compat.v1.AUTO_REUSE,
      )
      with dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 else dnn_scope:
        if self.bf16:
          net = tf.cast(net, dtype=tf.bfloat16)
        net = self._dnn(net, self._dnn_hidden_units, "hiddenlayer")
        if self.bf16:
          net = tf.cast(net, dtype=tf.float32)

        # dnn logits
        logits_scope = tf.compat.v1.variable_scope("logits")
        with logits_scope.keep_weights(
            dtype=tf.float32
        ) if self.bf16 else logits_scope as dnn_logits_scope:
          dnn_logits = tf.compat.v1.layers.dense(net,
                                                 units=1,
                                                 activation=None,
                                                 name=dnn_logits_scope)
          self._add_layer_summary(dnn_logits, dnn_logits_scope.name)

    # cross net
    with tf.compat.v1.variable_scope(
        "cross", partitioner=self._dense_layer_partitioner):
      # input layer
      with tf.compat.v1.variable_scope(
          "input_from_feature_columns",
          partitioner=self._input_layer_partitioner,
          reuse=tf.compat.v1.AUTO_REUSE,
      ):
        # Process categorical columns
        categorical_net_indexed = []
        for i in range(1, 27):
          col_name = "C" + str(i)
          indexed_column = tf.strings.to_hash_bucket_fast(
              self._feature[col_name], HASH_BUCKET_SIZES[col_name])
          # Embed categorical features
          dnn_embedding = get_kv_variable(
              name="embedding_weight_{}".format(i),
              key_dtype=tf.int64,
              embedding_dim=EMBEDDING_DIMENSIONS[col_name],
              initializer=tf.compat.v1.keras.initializers.RandomNormal(-1, 1),
          )
          embedded_column = embedding_lookup(
              params=dnn_embedding,
              ids=indexed_column,
              name="embedding_column_{}".format(i),
          )
          categorical_net_indexed.append(embedded_column)

        categorical_net = tf.concat(categorical_net_indexed, axis=-1)
        numerical_net = tf.compat.v1.feature_column.input_layer(
            features={
                numerical_column: self._feature[numerical_column]
                for numerical_column in CONTINUOUS_COLUMNS
            },
            feature_columns=[
                self._numerical_columns[numerical_column]
                for numerical_column in CONTINUOUS_COLUMNS
            ],
        )

        # Concatenate
        net = tf.concat([categorical_net, numerical_net], axis=-1)
        self._add_layer_summary(net, "input_from_feature_columns")
      # cross layers
      cross_scope = tf.compat.v1.variable_scope(
          "cross_layers",
          partitioner=self._dense_layer_partitioner,
          reuse=tf.compat.v1.AUTO_REUSE,
      )
      with cross_scope.keep_weights(
          dtype=tf.float32) if self.bf16 else cross_scope:
        if self.bf16:
          net = tf.cast(net, dtype=tf.bfloat16)

        net = self._cross_net(net, layer_name="crosslayer")

        if self.bf16:
          net = tf.cast(net, dtype=tf.float32)

        # cross logits
        logits_scope = tf.compat.v1.variable_scope("logits")
        with logits_scope.keep_weights(
            dtype=tf.float32
        ) if self.bf16 else logits_scope as cross_logits_scope:
          cross_logits = tf.compat.v1.layers.dense(net,
                                                   units=1,
                                                   activation=None,
                                                   name=cross_logits_scope)
          self._add_layer_summary(cross_logits, cross_logits_scope.name)

    self._logits = tf.add_n([dnn_logits, cross_logits])
    self.probability = tf.math.sigmoid(self._logits)
    self.output = tf.round(self.probability)

  def _create_loss(self):
    """ compute loss """
    self._logits = tf.squeeze(self._logits)
    self.loss = tf.compat.v1.losses.sigmoid_cross_entropy(
        self._label,
        self._logits,
        scope="loss",
        reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
    )
    tf.compat.v1.summary.scalar("loss", self.loss)

  def _create_optimizer(self):
    '''define optimizer and generate train_op'''
    self.global_step = tf.compat.v1.train.get_or_create_global_step()
    if self._optimizer_type == "adam":
      dnn_optimizer = AdamOptimizer(
          learning_rate=self._deep_learning_rate,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-8,
      )
      cross_optimizer = AdamOptimizer(
          learning_rate=self._cross_learning_rate,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-8,
      )
    elif self._optimizer_type == "adagrad":
      dnn_optimizer = AdagradOptimizer(
          learning_rate=0.1,
          initial_accumulator_value=0.1,
          use_locking=False,
      )
      cross_optimizer = AdagradOptimizer(
          learning_rate=0.1,
          initial_accumulator_value=0.1,
          use_locking=False,
      )
    elif self._optimizer_type == "group_adam":
      dnn_optimizer = GroupAdamOptimizer(
          learning_rate=self._cross_learning_rate,
          initial_accumulator_value=0.1,
          use_locking=False,
      )
      cross_optimizer = GroupAdamOptimizer(
          learning_rate=self._cross_learning_rate,
          initial_accumulator_value=0.1,
          use_locking=False,
      )
    elif self._optimizer_type == "sparse_group_ftrl":
      dnn_optimizer = SparseGroupFtrlOptimizer(
          learning_rate=0.1,
          initial_accumulator_value=0.1,
          use_locking=False,
      )
      cross_optimizer = SparseGroupFtrlOptimizer(
          learning_rate=0.1,
          initial_accumulator_value=0.1,
          use_locking=False,
      )
    else:
      raise ValueError("Optimizer type error.")

    train_ops = []
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_ops.append(
          dnn_optimizer.minimize(
              self.loss,
              var_list=tf.compat.v1.get_collection(
                  tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="dnn"),
              global_step=self.global_step,
          ))
      train_ops.append(
          cross_optimizer.minimize(
              self.loss,
              var_list=tf.compat.v1.get_collection(
                  tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                  scope="cross",
              ),
          ))
      self.train_op = tf.group(*train_ops)

  def _create_metrics(self):
    '''compute acc & auc'''
    self.acc, self.acc_op = tf.compat.v1.metrics.accuracy(
        labels=self._label, predictions=self.output)
    self.auc, self.auc_op = tf.compat.v1.metrics.auc(
        labels=self._label,
        predictions=self.probability,
        num_thresholds=200,
    )
    tf.compat.v1.summary.scalar("eval_acc", self.acc)
    tf.compat.v1.summary.scalar("eval_auc", self.auc)


def build_model_input(filename, batch_size, num_epochs):
  '''generate dataset pipline'''
  def parse_csv(value):
    tf.compat.v1.logging.info("Parsing {}".format(filename))
    cont_defaults = [[0.0] for i in range(1, 14)]
    cate_defaults = [[" "] for i in range(1, 27)]
    label_defaults = [[0]]
    column_headers = TRAIN_DATA_COLUMNS
    record_defaults = label_defaults + cont_defaults + cate_defaults
    columns = tf.io.decode_csv(value, record_defaults=record_defaults)
    all_columns = collections.OrderedDict(zip(column_headers, columns))
    labels = all_columns.pop(LABEL_COLUMN[0])
    features = all_columns
    return features, labels

  files = filename
  dataset = tf.data.TextLineDataset(files)
  dataset = dataset.shuffle(buffer_size=20000, seed=args.seed)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(parse_csv, num_parallel_calls=28)
  dataset = dataset.prefetch(2)
  return dataset


def build_feature_columns():
  '''generate feature columns'''
  # Notes: Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time.
  mins_list = [
      0.0,
      -3.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
  ]
  range_list = [
      1539.0,
      22069.0,
      65535.0,
      561.0,
      2655388.0,
      233523.0,
      26297.0,
      5106.0,
      24376.0,
      9.0,
      181.0,
      1807.0,
      6879.0,
  ]

  def make_minmaxscaler(min_value, range_value):

    def minmaxscaler(col):
      return (col - min_value) / range_value

    return minmaxscaler

  emb_stacking_categorical_columns = {}
  emb_stacking_numeric_columns = {}
  for column_name in FEATURE_COLUMNS:
    if column_name in CATEGORICAL_COLUMNS:
      categorical_column = \
        tf.compat.v1.feature_column.categorical_column_with_hash_bucket(
          column_name, hash_bucket_size=10000, dtype=tf.string)
      emb_stacking_categorical_columns[column_name] = categorical_column
    else:
      normalizer_fn = None
      i = CONTINUOUS_COLUMNS.index(column_name)
      normalizer_fn = make_minmaxscaler(mins_list[i], range_list[i])
      column = tf.compat.v1.feature_column.numeric_column(
          column_name, normalizer_fn=normalizer_fn, shape=(1, ))
      emb_stacking_numeric_columns[column_name] = column
  return [emb_stacking_categorical_columns, emb_stacking_numeric_columns]


def train(
    sess_config,
    input_hooks,
    model,
    data_init_op,
    steps,
    checkpoint_dir,
    tf_config=None,
    server=None,
):
  '''train func'''
  model.is_training = True
  hooks = []
  hooks.extend(input_hooks)

  scaffold = tf.compat.v1.train.Scaffold(
      local_init_op=tf.group(tf.compat.v1.local_variables_initializer(),
                             data_init_op),
      saver=tf.compat.v1.train.Saver(max_to_keep=args.keep_checkpoint_max),
  )

  stop_hook = tf.compat.v1.train.StopAtStepHook(last_step=steps)
  log_hook = tf.compat.v1.train.LoggingTensorHook(
      {
          "steps": model.global_step,
          "loss": model.loss
      }, every_n_iter=100)
  hooks.append(stop_hook)
  hooks.append(log_hook)
  if args.timeline > 0:
    hooks.append(
        tf.compat.v1.train.ProfilerHook(save_steps=args.timeline,
                                        output_dir=checkpoint_dir))
  # save_steps = args.save_steps if args.save_steps or args.no_eval else steps
  save_steps = steps

  with tf.compat.v1.train.MonitoredTrainingSession(
      master=server.target if server else "",
      is_chief=tf_config["is_chief"] if tf_config else True,
      hooks=hooks,
      scaffold=scaffold,
      checkpoint_dir=checkpoint_dir,
      save_checkpoint_steps=save_steps,
      summary_dir=checkpoint_dir,
      save_summaries_steps=args.save_steps,
      config=sess_config,
  ) as sess:
    while not sess.should_stop():
      sess.run([model.loss, model.train_op])
  print("Training completed.")

def evaluate(sess_config, input_hooks, model, data_init_op,
             steps, checkpoint_dir):
  '''evaluate func'''
  model.is_training = False
  hooks = []
  hooks.extend(input_hooks)

  scaffold = tf.compat.v1.train.Scaffold(local_init_op=tf.group(
      tf.compat.v1.local_variables_initializer(), data_init_op))
  session_creator = tf.compat.v1.train.ChiefSessionCreator(
      scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
  writer = tf.compat.v1.summary.FileWriter(os.path.join(checkpoint_dir, "eval"))
  merged = tf.compat.v1.summary.merge_all()
  print(merged)

  with tf.compat.v1.train.MonitoredSession(session_creator=session_creator,
                                           hooks=hooks) as sess:
    for step in range(1, steps + 1):
      if step != steps:
        sess.run([model.acc_op, model.auc_op])
        if step % 1000 == 0:
          print("Evaluation complete:[{}/{}]".format(step, steps))
      else:
        eval_acc, eval_auc, events = sess.run(
            [model.acc_op, model.auc_op, merged])
        writer.add_summary(events, step)
        print("Evaluation complete:[{}/{}]".format(step, steps))
        print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))


def main(tf_config=None, server=None):
  # check dataset and count data set size
  print("Checking dataset...")
  train_file = args.data_location
  test_file = args.data_location

  train_file += "/train.csv"
  test_file += "/eval.csv"

  if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
    print("Dataset does not exist in the given data_location.")
    sys.exit()

  # pylint: disable=consider-using-with
  no_of_training_examples = sum(1 for line in open(train_file,
                                                    encoding='utf-8'))
  # pylint: disable=consider-using-with
  no_of_test_examples = sum(1 for line in open(test_file, encoding='utf-8'))
  print("Numbers of training dataset is {}".format(no_of_training_examples))
  print("Numbers of test dataset is {}".format(no_of_test_examples))

  # set batch size, epoch & steps
  batch_size = args.batch_size

  if args.steps == 0:
    no_of_epochs = 1
    train_steps = math.ceil(
        (float(no_of_epochs) * no_of_training_examples) / batch_size)
  else:
    no_of_epochs = math.ceil(
        (float(batch_size) * args.steps) / no_of_training_examples)
    train_steps = args.steps
  test_steps = math.ceil(float(no_of_test_examples) / batch_size)
  print("The training steps is {}".format(train_steps))
  print("The testing steps is {}".format(test_steps))

  # set fixed random seed
  tf.random.set_seed(args.seed)

  # set directory path for checkpoint_dir
  model_dir = os.path.join(args.output_dir,
                           "model_DCN_" + str(int(time.time())))

  checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
  print("Saving model checkpoints to " + checkpoint_dir)

  # create data pipline of train & test dataset
  train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
  test_dataset = build_model_input(test_file, batch_size, 1)

  dataset_output_types = tf.compat.v1.data.get_output_types(train_dataset)
  dataset_output_shapes = tf.compat.v1.data.get_output_shapes(test_dataset)
  iterator = tf.compat.v1.data.Iterator.from_structure(dataset_output_types,
                                                       dataset_output_shapes)

  train_init_op = iterator.make_initializer(train_dataset)
  test_init_op = iterator.make_initializer(test_dataset)

  next_element = iterator.get_next()

  # create feature column
  emb_stacking_columns = build_feature_columns()

  input_layer_partitioner = None
  dense_layer_partitioner = None

  # Session config
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.inter_op_parallelism_threads = args.inter
  sess_config.intra_op_parallelism_threads = args.intra

  hooks = []

  # create model
  model = DCN(
      emb_stacking_columns=emb_stacking_columns,
      # linear_learning_rate=args.linear_learning_rate,
      deep_learning_rate=args.deep_learning_rate,
      optimizer_type=args.optimizer,
      bf16=args.bf16,
      stock_tf=args.tf,
      inputs=next_element,
      input_layer_partitioner=input_layer_partitioner,
      dense_layer_partitioner=dense_layer_partitioner,
  )
  kv_variable_ops.IS_TRAINING = True
  # Run model training and evaluation
  train(
      sess_config,
      hooks,
      model,
      train_init_op,
      train_steps,
      checkpoint_dir,
      tf_config,
      server,
  )
  kv_variable_ops.IS_TRAINING = False
  if not (args.no_eval or tf_config):
    evaluate(sess_config, hooks, model, test_init_op,
             test_steps, checkpoint_dir)


def boolean_string(string):
  low_string = string.lower()
  if low_string not in {"false", "true"}:
    raise ValueError("Not a valid boolean string")
  return low_string == "true"


def get_arg_parser():
  '''get parse'''
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_location",
      help="Full path of train data",
      required=False,
      default="./data",
  )
  parser.add_argument(
      "--steps",
      help="set the number of steps on train dataset",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--batch_size",
      help="Batch size to train. Default is 2048",
      type=int,
      default=2048,
  )
  parser.add_argument(
      "--output_dir",
      help="Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. ",
      required=False,
      default="./result",
  )
  parser.add_argument(
      "--checkpoint",
      help="Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP",
      required=False,
  )
  parser.add_argument(
      "--save_steps",
      help="set the number of steps on saving checkpoints",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--seed",
      help="set the random seed for tensorflow",
      type=int,
      default=2021,
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      choices=["adam", "adagrad", "group_adam", "sparse_group_ftrl"],
      default="adam",
  )
  parser.add_argument(
      "--deep_learning_rate",
      help="Learning rate for deep model",
      type=float,
      default=0.01,
  )
  parser.add_argument(
      "--keep_checkpoint_max",
      help="Maximum number of recent checkpoint to keep",
      type=int,
      default=1,
  )
  parser.add_argument(
      "--timeline",
      help="number of steps on saving timeline. Default 0",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--inter",
      help="set inter op parallelism threads.",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--intra",
      help="set inter op parallelism threads.",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--bf16",
      help="enable DeepRec BF16 in deep model. Default FP32",
      action="store_true",
  )
  parser.add_argument(
      "--no_eval",
      help="not evaluate trained model by eval dataset.",
      action="store_true",
  )
  parser.add_argument(
      "--tf",
      help="Use TF 2.13.0 API and disable tfplus feature to run a baseline.",
      action="store_true",
  )

  return parser


if __name__ == "__main__":
  parser_ = get_arg_parser()
  args = parser_.parse_args()
  main()
