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

import time

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.training import monitored_session, session_manager
from tensorflow.python.training.monitored_session import _WrappedSession
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys

from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.tensorflow.util.tf_version_util import (
    is_tf_2,
    is_tf_113,
    is_tf_115,
)
from dlrover.trainer.util.log_util import default_logger as logger


def wait_for_session_and_get_session(
    self, master, config=None, max_wait_secs=float("Inf")
):
    """Creates a new `Session` and waits for model to be ready.

    Creates a new `Session` on 'master'.  Waits for the model to be
    initialized or recovered from a checkpoint.  It's expected that
    another thread or process will make the model ready, and that this
    is intended to be used by threads/processes that participate in a
    distributed training configuration where a different thread/process
    is responsible for initializing or recovering the model being trained.

    NB: The amount of time this method waits for the session is bounded
    by max_wait_secs. By default, this function will wait indefinitely.

    Args:
      master: `String` representation of the TensorFlow master to use.
      config: Optional ConfigProto proto used to configure the session.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      A `Session`. May be None if the operation exceeds the timeout
      specified by config.operation_timeout_in_ms.

    Raises:
      tf.DeadlineExceededError: if the session is not available after
        max_wait_secs.
    """
    global_dict = common_util.GlobalDict()
    if "session_creation_count" not in global_dict:
        global_dict["session_creation_count"] = 0
    global_dict["session_creation_count"] += 1

    self._target = master

    if max_wait_secs is None:
        max_wait_secs = float("Inf")
    timer = session_manager._CountDownTimer(max_wait_secs)

    while True:
        sess = session.Session(self._target, graph=self._graph, config=config)
        not_ready_msg = None
        not_ready_local_msg = None
        local_init_success, not_ready_local_msg = self._try_run_local_init_op(
            sess
        )
        if local_init_success:
            # Successful if local_init_op is None, or ready_for_local_init_op passes # noqa: E501
            is_ready, not_ready_msg = self._model_ready(sess)
            if is_ready:
                global_dict = common_util.GlobalDict()
                global_dict["sess"] = sess
                return sess

        self._safe_close(sess)

        # Do we have enough time left to try again?
        remaining_ms_after_wait = (
            timer.secs_remaining() - self._recovery_wait_secs
        )
        if remaining_ms_after_wait < 0:
            raise errors.DeadlineExceededError(
                None,
                None,
                "Session was not ready after waiting %d secs."
                % (max_wait_secs,),
            )

        logger.info(
            "Waiting for model to be ready.  "
            "Ready_for_local_init_op:  %s, ready: %s",
            not_ready_local_msg,
            not_ready_msg,
        )
        time.sleep(self._recovery_wait_secs)


def init_and_get_session_creator(self, sess_creator):
    """Create a new `_RecoverableSession`.

    The value returned by calling `sess_creator.create_session()` will be the
    session wrapped by this recoverable session.

    Args:
        sess_creator: A 'SessionCreator' to be wrapped by recoverable.
    """
    self._sess_creator = sess_creator
    _WrappedSession.__init__(self, self._create_session())
    global_dict = common_util.GlobalDict()
    logger.info("self._sess_creator type {}".format(type(self._sess_creator)))
    logger.info("self._sess_creator {}".format(dir(self._sess_creator)))
    logger.info(
        "self._sess_creator._session_creator._config {}".format(
            str(self._sess_creator._session_creator._config)
        )
    )
    global_dict["session_creator"] = self._sess_creator._session_creator


def prepare_session_113(
    self,
    master,
    init_op=None,
    saver=None,
    checkpoint_dir=None,
    checkpoint_filename_with_path=None,
    wait_for_checkpoint=False,
    max_wait_secs=7200,
    config=None,
    init_feed_dict=None,
    init_fn=None,
):
    """Creates a `Session`. Makes sure the model is ready to be used.

    Creates a `Session` on 'master'. If a `saver` object is passed in, and
    `checkpoint_dir` points to a directory containing valid checkpoint
    files, then it will try to recover the model from checkpoint. If
    no checkpoint files are available, and `wait_for_checkpoint` is
    `True`, then the process would check every `recovery_wait_secs`,
    up to `max_wait_secs`, for recovery to succeed.

    If the model cannot be recovered successfully then it is initialized by
    running the `init_op` and calling `init_fn` if they are provided.
    The `local_init_op` is also run after init_op and init_fn, regardless of
    whether the model was recovered successfully, but only if
    `ready_for_local_init_op` passes.

    If the model is recovered from a checkpoint it is assumed that all
    global variables have been initialized, in particular neither `init_op`
    nor `init_fn` will be executed.

    It is an error if the model cannot be recovered and no `init_op`
    or `init_fn` or `local_init_op` are passed.

    Args:
      master: `String` representation of the TensorFlow master to use.
      init_op: Optional `Operation` used to initialize the model.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the # noqa: E501
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file. # noqa: E501
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.
      init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
        values.  This feed dictionary is passed to the session `run()` call when # noqa: E501
        running the init op.
      init_fn: Optional callable used to initialize the model. Called after the
        optional `init_op` is called.  The callable must accept one argument,
        the session being initialized.

    Returns:
      A `Session` object that can be used to drive the model.

    Raises:
      RuntimeError: If the model cannot be initialized or recovered.
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """

    sess, is_loaded_from_checkpoint = self._restore_checkpoint(
        master,
        saver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename_with_path=checkpoint_filename_with_path,
        wait_for_checkpoint=wait_for_checkpoint,
        max_wait_secs=max_wait_secs,
        config=config,
    )
    if not is_loaded_from_checkpoint:
        if init_op is None and not init_fn and self._local_init_op is None:
            raise RuntimeError(
                "Model is not initialized and no init_op or "
                "init_fn or local_init_op was given"
            )
        if init_op is not None:
            sess.run(init_op, feed_dict=init_feed_dict)
        if init_fn:
            init_fn(sess)

    local_init_success, msg = self._try_run_local_init_op(sess)
    if not local_init_success:
        raise RuntimeError(
            "Init operations did not make model ready for local_init.  "
            "Init op: %s, init fn: %s, error: %s"
            % (session_manager._maybe_name(init_op), init_fn, msg)
        )

    is_ready, msg = self._model_ready(sess)
    if not is_ready:
        raise RuntimeError(
            "Init operations did not make model ready.  "
            "Init op: %s, init fn: %s, local_init_op: %s, error: %s"
            % (
                session_manager._maybe_name(init_op),
                init_fn,
                self._local_init_op,
                msg,
            )
        )
    global_dict = common_util.GlobalDict()
    global_dict["sess"] = sess
    return sess


def prepare_session_115(
    self,
    master,
    init_op=None,
    saver=None,
    checkpoint_dir=None,
    checkpoint_filename_with_path=None,
    wait_for_checkpoint=False,
    max_wait_secs=7200,
    config=None,
    init_feed_dict=None,
    init_fn=None,
    incr_saver=None,
):
    """Creates a `Session`. Makes sure the model is ready to be used.

    Creates a `Session` on 'master'. If a `saver` object is passed in, and
    `checkpoint_dir` points to a directory containing valid checkpoint
    files, then it will try to recover the model from checkpoint. If
    no checkpoint files are available, and `wait_for_checkpoint` is
    `True`, then the process would check every `recovery_wait_secs`,
    up to `max_wait_secs`, for recovery to succeed.

    If the model cannot be recovered successfully then it is initialized by
    running the `init_op` and calling `init_fn` if they are provided.
    The `local_init_op` is also run after init_op and init_fn, regardless of
    whether the model was recovered successfully, but only if
    `ready_for_local_init_op` passes.

    If the model is recovered from a checkpoint it is assumed that all
    global variables have been initialized, in particular neither `init_op`
    nor `init_fn` will be executed.

    It is an error if the model cannot be recovered and no `init_op`
    or `init_fn` or `local_init_op` are passed.

    Args:
      master: `String` representation of the TensorFlow master to use.
      init_op: Optional `Operation` used to initialize the model.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the # noqa: E501
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file. # noqa: E501
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.
      init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
        values.  This feed dictionary is passed to the session `run()` call when # noqa: E501
        running the init op.
      init_fn: Optional callable used to initialize the model. Called after the
        optional `init_op` is called.  The callable must accept one argument,
        the session being initialized.

    Returns:
      A `Session` object that can be used to drive the model.

    Raises:
      RuntimeError: If the model cannot be initialized or recovered.
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """

    sess, is_loaded_from_checkpoint = self._restore_checkpoint(
        master,
        saver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename_with_path=checkpoint_filename_with_path,
        wait_for_checkpoint=wait_for_checkpoint,
        max_wait_secs=max_wait_secs,
        config=config,
    )
    if not is_loaded_from_checkpoint:
        if init_op is None and not init_fn and self._local_init_op is None:
            raise RuntimeError(
                "Model is not initialized and no init_op or "
                "init_fn or local_init_op was given"
            )
        if init_op is not None:
            sess.run(init_op, feed_dict=init_feed_dict)
        if init_fn:
            init_fn(sess)

    local_init_success, msg = self._try_run_local_init_op(sess)
    if not local_init_success:
        raise RuntimeError(
            "Init operations did not make model ready for local_init.  "
            "Init op: %s, init fn: %s, error: %s"
            % (session_manager._maybe_name(init_op), init_fn, msg)
        )

    is_ready, msg = self._model_ready(sess)
    if not is_ready:
        raise RuntimeError(
            "Init operations did not make model ready.  "
            "Init op: %s, init fn: %s, local_init_op: %s, error: %s"
            % (
                session_manager._maybe_name(init_op),
                init_fn,
                self._local_init_op,
                msg,
            )
        )
    global_dict = common_util.GlobalDict()
    global_dict["sess"] = sess
    return sess


def export_saved_model(
    self,
    export_dir_base,
    serving_input_receiver_fn,
    assets_extra=None,
    as_text=False,
    checkpoint_path=None,
    experimental_mode=ModeKeys.PREDICT,
    save_incr_model=True,
):
    # pylint: enable=line-too-long
    if not serving_input_receiver_fn:
        raise ValueError("An input_receiver_fn must be defined.")

    input_receiver_fn_map = {experimental_mode: serving_input_receiver_fn}

    return self._export_all_saved_models(
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=True,
        save_incr_model=save_incr_model,
    )


def hotpatch_for_dynet(failover_level=1):
    """Patch for tensorflow in order to"""

    logger.info("Hot patch for dynet")
    if failover_level == 1:
        # Get the session after initialization
        monitored_session._RecoverableSession.__init__ = (
            init_and_get_session_creator
        )
        session_manager.SessionManager.wait_for_session = (
            wait_for_session_and_get_session
        )
    if is_tf_115():
        session_manager.SessionManager.prepare_session = prepare_session_115

    if is_tf_113() or is_tf_2():
        session_manager.SessionManager.prepare_session = prepare_session_113
