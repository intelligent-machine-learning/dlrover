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
from tensorflow.python.training.basic_session_run_hooks import (
    SecondOrStepTimer,
)
from tensorflow.python.training.session_run_hook import SessionRunHook

from dlrover.trainer.util.log_util import default_logger as logger

tf.disable_v2_behavior()


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
