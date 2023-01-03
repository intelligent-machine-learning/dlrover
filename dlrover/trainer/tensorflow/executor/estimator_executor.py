from dlrover.constants.tf_constants import TFConstants
from dlrover.util.log_util import default_logger as logger
from dlrover.tensorflow.util.dataset_util import DatasetUtil
from dlrover.util.reflect_util import get_class
from dlrover.tensorflow.util.estimator_util import hook_estimator_call_model_fn
from dlrover.tensorflow.hooks.global_step_hook import GlobalStepHook
from tensorflow.python.training.session_run_hook import SessionRunHook
from dlrover.tensorflow.executor.base_executor import BaseExecutor
import tensorflow as tf
import os

try:
    from dlrover.python.elastic_agent.tensorflow.hooks import (
        ElasticDataShardReportHook,
        ReportModelMetricHook,
    )
except Exception as e:
    logger.warning("fail to import dlrover")

class EstimatorExecutor(BaseExecutor):
    def __init__(self, context, can_pickle=False, context_from_storage=False):
        """
        Args:
            context: penrose.context.ExecutorContext
            unpickle: need pickle.unpickle(context) or not
            context_from_storage: We saved the `context` value in storage and
                                passed the storage-key as the `context`
        """

        super(EstimatorExecutor).__init__() 
        self.get_cluster_info_by_tf_config()

        
        self._initialize_estimator_related()
        self._prepare_env()
        # prepare estimator class from user
        self.gen_model_dir()
        self._task_conf = context
        self._prepare_estimator_class()
        self._prepare_estimator()



    def gen_model_dir(self):
        self._model_dir = "/input/model/test1"
        if not os.path.exists(self._model_dir):
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
        + export_model_context: a dict that matches:
            penrose.util.internal.model_export_util.ModelExporter.export_model
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
            " penrose will decide the training and evaluting process."
        )
        if isinstance(classifier_class, str):
            self._classifier_class = get_class(classifier_class)
        else:
            self._classifier_class = classifier_class

        self._prepare_dataset()
        self._prepare_train_spec()
        self._prepare_eval_spec()
        
        config, params = self._prepare_estimator_config_and_params()
        self._tf_estimator = self._classifier_class(
            self._model_dir, config, params
        )


    def _prepare_estimator_config_and_params(self):
        session_config = self._task_conf.get(
            TFConstants.SessionConfig.name, None
        )
        run_config = self._task_conf.get(TFConstants.RunConfig.name, {})
        
        config = self.get_config(self.cluster_spec)
        config._model_dir=self._model_dir
        params = {}
        log_steps = self._task_conf.get(
            TFConstants.LogSteps.name, TFConstants.LogSteps()
        )
        training_hooks = [GlobalStepHook()]
        data_shard_client = self.train_dataset.reader.data_shard_client
        if data_shard_client is not None:
            logger.info("appending ElasticDataShardReportHook")
            class ElasticDataShardReportHook(SessionRunHook):
                def __init__(self, sharding_client):
                    self._sharding_client = sharding_client

                def after_run(self, run_context, run_values):
                    try:
                        self._sharding_client.report_batch_done()
                        logger.info("report_batch_done")
                    except Exception as ex:
                        logger.error("DLrover agent: report batch done failed: %s", ex)

            shard_report_hook = ElasticDataShardReportHook(data_shard_client)
            #model_metric_report_hook = ReportModelMetricHook()
            training_hooks.append(shard_report_hook)
            #training_hooks.append(model_metric_report_hook)
        params[TFConstants.EstimatorTrainingHooks.name] = training_hooks
        hook_estimator_call_model_fn(params)
        user_params = {}
        logger.info("config is {}".format(config))
        return config, user_params

    def _prepare_dataset(self):
        train_set = self._task_conf.get("train_set")
        self.train_dataset = DatasetUtil.create(train_set)

    def _prepare_train_input_fn(self):
        self._train_input_fn = self.train_dataset.input_fn()
        return self._train_input_fn

    def _prepare_eval_input_fn(self):
        self._eval_input_fn = self.train_dataset.input_fn()
        return self._eval_input_fn

    def _prepare_train_spec(self):
        self._prepare_train_input_fn()
        max_steps = self._task_conf.get(TFConstants.MaxSteps.name, None)
        input_fn = self._train_input_fn
        self._train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn, max_steps=max_steps,
        )

    def _prepare_eval_spec(self):
        self._prepare_eval_input_fn()
        steps = 100
        self._eval_spec = tf.estimator.EvalSpec(
            input_fn=self._eval_input_fn,
            steps=steps,
            throttle_secs=10,
            start_delay_secs=5,
        )
    
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