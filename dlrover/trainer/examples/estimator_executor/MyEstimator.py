from dlrover.trainer.util.log_util import default_logger as logger
from dlrover.trainer.tensorflow.util.dataset_util import DatasetUtil
from dlrover.trainer.tensorflow.util.column_info import Column
import tensorflow as tf
from dlrover.trainer.tensorflow.hooks.global_step_hook import GlobalStepHook

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

class MyEstimator(tf.estimator.Estimator):
    """MyEstimator"""

    def __init__(self, model_dir, config=None, params=None):

        logger.info("buildinf model fn")
        logger.info("config is {}".format(config))
        import pdb 
        #pdb.set_trace()
        #run_config = tf.estimator.RunConfig(model_dir=model_dir)
        run_config = config

        super(MyEstimator, self).__init__(
            self.model_fn,
            model_dir=model_dir,
            config=run_config,
            params=params,
        )

    def model_fn(self, features, labels, mode, params):
        # 具体的含义见
        # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#args
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
