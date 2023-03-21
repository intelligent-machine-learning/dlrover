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

from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.training import basic_session_run_hooks

from dlrover.python.elastic_agent.sychronization.sync_client import SyncClient
from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.tensorflow.util.tf_env_util import (
    get_tf_config_task_type_and_index,
)
from dlrover.trainer.util.log_util import default_logger as logger


def after_run(self, run_context, run_values):
    task_type, _ = get_tf_config_task_type_and_index()
    global_dict = common_util.GlobalDict()
    relaunch_for_ps = global_dict.get(
        TFConstants.RelaunchForPs.name, TFConstants.RelaunchForPs()
    )
    if relaunch_for_ps:
        logger.info(
            "The training thread should stop for due to ps migration/scaling"
        )
    if relaunch_for_ps:
        # Only worker need to wait for cheif to do somethin before exit
        # Chief doesn't need to wait.
        if task_type == TFConstants.Worker.name:
            SyncClient().join_sync("relauch_for_ps")
            logger.info(
                "Before stopping training thread,  \
                worker should wait for cheif to save checkpoint"
            )
        run_context.request_stop()
        if task_type == TFConstants.Worker.name:
            # Workers need to wait for cheif to do somethin before exit
            SyncClient().barrier("relauch_for_ps")
            logger.info(
                "Training thread stopped because chief had saved checkpoint"
            )
        else:
            # Chief need to nofity workers
            SyncClient().notify_barrier("relauch_for_ps")
            logger.info(
                "Checkpointed saved, cheif notify \
                workers that they can stop training thread."
            )
            fail_over = global_dict["failover"]
            fail_over._failover_client.ready_for_ps_relaunch()


basic_session_run_hooks.StopAtStepHook.after_run = after_run


def ck_after_run(self, run_context, run_values):
    stale_global_step = run_values.results
    global_dict = common_util.GlobalDict()
    should_save_checkpoint = global_dict.get(
        TFConstants.SaveCheckpoint.name, TFConstants.SaveCheckpoint()
    )
    if should_save_checkpoint:
        logger.info(
            "Before saving checkpoint, cheif should wait for \
                worker to enter PreStopAtStep Hook."
        )
        # CheckpointSaveHook is a kind of chiefhook
        # Only chief run the hook
        SyncClient().join_sync("relauch_for_ps")
        logger.info(
            "All workers have entered PreStopAtStep Hook \
                and wait for cheif to save checkpoints"
        )
        # chief can relaun ps
        global_dict[TFConstants.RelaunchForPs.name] = True
    if (
        self._timer.should_trigger_for_step(
            stale_global_step + self._steps_per_run
        )
        or should_save_checkpoint
    ):
        # get the real value after train op.
        global_step = run_context.session.run(self._global_step_tensor)
        if (
            self._timer.should_trigger_for_step(global_step)
            or should_save_checkpoint
        ):
            self._timer.update_last_triggered_step(global_step)
            if self._save(run_context.session, global_step):
                run_context.request_stop()
    elif self._incremental_save:
        if (
            self._incremental_timer.should_trigger_for_step(
                stale_global_step + 1
            )
            or should_save_checkpoint
        ):
            global_step = run_context.session.run(self._global_step_tensor)
            if (
                self._incremental_timer.should_trigger_for_step(global_step)
                or should_save_checkpoint
            ):
                self._incremental_timer.update_last_triggered_step(global_step)
                logger.info(
                    "Start Save incremental checkpoints for %d into %s.",
                    global_step,
                    self._incremental_save_path,
                )
                self._get_incr_saver().incremental_save(
                    run_context.session,
                    self._incremental_save_path,
                    global_step=global_step,
                )
                logger.info(
                    "Finish Save incremental checkpoints for %d into %s.",
                    global_step,
                    self._incremental_save_path,
                )


def append_hooks(estimator_spec, key, params):
    old = getattr(estimator_spec, key) or []
    hooks = [hook for hook in params.get(key, [])]
    if hooks:
        hooks_names = [c.__class__.__name__ for c in hooks]
        hooks.extend(old)
        logger.info("Hooks before deduplication: %s = %s", key, hooks_names)

        def _unique(hooks):
            dup = dict()
            result = []
            for h in hooks:
                name = h.__class__.__name__
                if name not in dup:
                    result.append(h)
                else:
                    logger.warning(
                        "%s has existed, it won't be added",
                        h.__class__.__name__,
                    )
                dup[name] = h
            return result

        hooks = _unique(hooks)
        logger.info(
            "Appending hooks after deduplication: %s = %s",
            key,
            [c.__class__.__name__ for c in hooks],
        )
        return estimator_spec._replace(**{key: hooks})
    else:
        return estimator_spec


def hook_estimator_call_model_fn(params=None):

    estimator_call_model_fn = Estimator._call_model_fn

    def dlrover_call_model_fn(*args, **kwargs):
        model_fn_results = estimator_call_model_fn(*args, **kwargs)
        if params:
            keys = [
                TFConstants.EstimatorTrainingChiefHooks.name,
                TFConstants.EstimatorTrainingHooks.name,
                TFConstants.EstimatorEvaluationHooks.name,
                TFConstants.EstimatorPredictionHooks.name,
            ]
            stop_at_step_hook = basic_session_run_hooks.StopAtStepHook(
                num_steps=10
            )
            training_hooks = params.get(
                TFConstants.EstimatorTrainingHooks.name, []
            )
            training_hooks.append(stop_at_step_hook)
            params[TFConstants.EstimatorTrainingHooks.name] = training_hooks

            chief_training_hooks = params.get(
                TFConstants.EstimatorTrainingChiefHooks.name, []
            )
            chief_training_hooks.append(stop_at_step_hook)
            params[
                TFConstants.EstimatorTrainingChiefHooks.name
            ] = chief_training_hooks

            for key in keys:
                model_fn_results = append_hooks(model_fn_results, key, params)
        return model_fn_results

    Estimator._call_model_fn = dlrover_call_model_fn
