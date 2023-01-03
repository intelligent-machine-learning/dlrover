import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import (
    SecondOrStepTimer,
)
from dlrover.trainer.util.log_util import default_logger as logger
import time


class GlobalStepHook(SessionRunHook):
    def __init__(self, every_n_iter=1):
        self._fetches = dict()
        self._timer = SecondOrStepTimer(every_steps=every_n_iter)
        logger.info("ModelSizeHook: every_n_iter: {}".format(every_n_iter))

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        self._fetches["global_step"] = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        """before_run"""
        session = run_context.session
        global_step = session.run(self._fetches["global_step"])
        logger.info("global_step: {}".format(global_step))


    def end(self, session):
        logger.info("hook end")
