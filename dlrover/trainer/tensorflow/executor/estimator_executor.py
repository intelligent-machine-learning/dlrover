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
import time

import tensorflow as tf
from tensorflow.python.estimator.exporter import BestExporter
from tensorflow.python.ops import array_ops
from tensorflow.python.training.basic_session_run_hooks import (
    CheckpointSaverHook,
)
from tensorflow_estimator.python.estimator.estimator import Estimator

from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.executor.base_executor import BaseExecutor
from dlrover.trainer.tensorflow.hooks.elastic_data_shard_report_hook import (
    ElasticDataShardReportHook,
)
from dlrover.trainer.tensorflow.hooks.global_step_hook import GlobalStepHook
from dlrover.trainer.tensorflow.util.data_mapping_util import data_mapping
from dlrover.trainer.tensorflow.util.dataset_util import DatasetUtil
from dlrover.trainer.tensorflow.util.estimator_util import (
    ck_after_run,
    hook_estimator_call_model_fn,
)
from dlrover.trainer.tensorflow.util.tf_patch_util import export_saved_model
from dlrover.trainer.tensorflow.util.tf_version_util import is_tf_115
from dlrover.trainer.util.log_util import default_logger as logger
from dlrover.trainer.util.reflect_util import get_class

try:
    from dlrover.python.elastic_agent.tensorflow.hooks import (
        ReportModelMetricHook,
    )
except Exception:
    logger.warning("fail to import dlrover")


class EstimatorExecutor(BaseExecutor):
    """
    EstimatorExecutor is a wrapper for tensorflow estimator.
    It helps prepare estimator which speicified in the config
    by relcecting method, parse inputs, build train_spec, eval_spec.
    """

    def __init__(self, context, context_from_storage=False):
        """
        Args:
            context_from_storage: We saved the `context` value in storage and
                                passed the storage-key as the `context`
        """

        super(EstimatorExecutor, self).__init__()
        self._task_conf = context

    def wait_for_tf_config(self):
        while os.environ.get("TF_CONFIG", None) is None:
            time.sleep(1)

    def set_tf_config(self, tf_config):
        if not isinstance(tf_config, str):
            tf_config = json.dumps(tf_config)
        os.environ["TF_CONFIG"] = tf_config
        self.prepare()

    def prepare(self):

        self.get_cluster_info_by_tf_config()
        self._initialize_estimator_related()
        self._prepare_env()

        self._prepare_estimator_class()
        self._prepare_estimator()
        self._prepare_incr_saved_model_checkpoint()

    def gen_model_dir(self):
        self._model_dir = self._task_conf.get(TFConstants.ModelDir.name)
        if (
            not os.path.exists(self._model_dir)
            and self.task_type == TFConstants.Chief()
        ):
            os.makedirs(self._model_dir)

    def _initialize_estimator_related(self):
        pass

    def _prepare_env(self):
        pass

    def _prepare_estimator_class(self):
        pass

    @property
    def reader(self):
        pass

    def _get_classifier(self):
        """Get valid classifier instance"""

        classifier_class = self._task_conf.get(
            TFConstants.EstimatorClassifierClass.name, None
        )
        return classifier_class

    def _prepare_estimator(self, need_context=False):
        """Preparation for estimator
        + EstimatorSpec
        + TrainSpec
        + Estimator
        """
        classifier_class = self._get_classifier()
        if classifier_class is None:
            logger.warning(
                "No classifier_class function definition found in"
                " the estimator class you provided, will call"
                " your `run` directly"
            )
            return
        logger.info(
            "found `classifier_class` in your estimator class"
            " dlrover.trainer will decide the training and evaluting process."
        )
        if isinstance(classifier_class, str):
            self._classifier_class = get_class(classifier_class)
        else:
            self._classifier_class = classifier_class

        self._prepare_train_dataset()
        self._prepare_eval_dataset()
        self._prepare_train_spec()
        self._prepare_eval_spec()

        config, params = self._prepare_estimator_config_and_params()
        conf_params = self._task_conf.get("params", {})
        params.update(conf_params)
        self._tf_estimator = self._classifier_class(
            self._model_dir, config, params
        )

    def _prepare_estimator_config_and_params(self):
        """prepare estimator.RunConfig and set default estimator hooks"""
        config = self.get_config(self.cluster_spec)
        self.gen_model_dir()
        config._model_dir = self._model_dir
        config._keep_checkpoint_max = self._task_conf.get(
            TFConstants.KeepCheckpointMax.name, TFConstants.KeepCheckpointMax()
        )
        params = {}
        training_hooks = [GlobalStepHook()]
        data_shard_client = self.train_dataset.reader.data_shard_client

        if data_shard_client is not None:
            logger.info("appending ElasticDataShardReportHook")
            shard_report_hook = ElasticDataShardReportHook(data_shard_client)
            model_metric_report_hook = ReportModelMetricHook()
            training_hooks.append(shard_report_hook)
            training_hooks.append(model_metric_report_hook)

        params[TFConstants.EstimatorTrainingHooks.name] = training_hooks

        save_steps = self._task_conf.get(
            TFConstants.SaveSteps.name, TFConstants.SaveSteps()
        )
        save_secs = self._task_conf.get(
            TFConstants.SaveSecs.name, TFConstants.SaveSecs()
        )
        logger.info("checkpoint hook %s", self._model_dir)
        checkpoint_save_hook = CheckpointSaverHook(
            self._model_dir, save_steps=save_steps, save_secs=save_secs
        )
        checkpoint_incremental_save_secs = self._task_conf.get(
            TFConstants.CheckpointIncrementalSaveSecs.name, None
        )
        if is_tf_115() and checkpoint_incremental_save_secs:
            CheckpointSaverHook.after_run = ck_after_run
            checkpoint_save_hook = CheckpointSaverHook(
                self._model_dir,
                save_steps=save_steps,
                save_secs=save_secs,
                incremental_save_secs=checkpoint_incremental_save_secs,
            )

        params[TFConstants.EstimatorTrainingChiefHooks.name] = [
            checkpoint_save_hook
        ]
        train_set = self._task_conf.get(TFConstants.TrainSet.name)
        params["columns"] = train_set.columns
        hook_estimator_call_model_fn(params)
        user_params = {}
        logger.info("config is {}".format(config))
        return config, user_params

    def _prepare_train_dataset(self):
        """prepare_train_dataset"""
        train_set = self._task_conf.get(TFConstants.TrainSet.name)
        self.train_dataset = DatasetUtil.create(train_set)

    def _prepare_eval_dataset(self):
        """prepare_eval_datasets"""
        eval_set = self._task_conf.get(TFConstants.EvalSet.name)
        self.eval_dataset = DatasetUtil.create(eval_set)

    def _prepare_train_input_fn(self):
        self._train_input_fn = self.train_dataset.input_fn()
        return self._train_input_fn

    def _prepare_eval_input_fn(self):
        self._eval_input_fn = self.eval_dataset.input_fn()
        return self._eval_input_fn

    def _prepare_train_spec(self):
        self._prepare_train_input_fn()
        max_steps = self._task_conf.get(
            TFConstants.MaxSteps.name, TFConstants.MaxSteps()
        )
        input_fn = self._train_input_fn
        self._train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn,
            max_steps=max_steps,
        )

    def _prepare_eval_spec(self):
        self._prepare_eval_input_fn()
        eval_steps = self._task_conf.get(
            TFConstants.EvalSteps.name, TFConstants.EvalSteps()
        )
        columns = self.eval_dataset.columns

        def serving_input_receiver_fn():
            feature_map = {}
            for i in columns:
                # Todo: support different shape
                feature_map[i.name] = array_ops.placeholder(
                    dtype=data_mapping[i.dtype], shape=[None]
                )
            return tf.estimator.export.build_raw_serving_input_receiver_fn(
                feature_map
            )

        exporter = BestExporter(
            serving_input_receiver_fn=serving_input_receiver_fn()
        )
        logger.info("Adding export to estimator_spec")

        self._eval_spec = tf.estimator.EvalSpec(
            input_fn=self._eval_input_fn,
            steps=eval_steps,
            throttle_secs=10,
            start_delay_secs=5,
            exporters=[exporter],
        )

    def _prepare_incr_saved_model_checkpoint(self):
        # check whether tf is deeprec
        if is_tf_115():
            Estimator.export_saved_model = export_saved_model

    def train_and_evaluate(self):
        logger.info("starting train and evaluate")
        try:
            tf.estimator.train_and_evaluate(
                self._tf_estimator, self._train_spec, self._eval_spec
            )
        except tf.errors.OutOfRangeError:
            logger.warn(
                "No more data, stop early"
                ", please check batch_size"
                "* steps is less than total"
                "count of training/testing set,"
                "or the data volume is expected."
            )
