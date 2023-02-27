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
from tensorflow_estimator.python.estimator import early_stopping

from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.util import common_util
from dlrover.trainer.util.log_util import default_logger as logger


def after_run(self, run_context, run_values):
    global_step = run_values.results + 1
    if global_step >= self._last_step:

        step = run_context.session.run(self._global_step_tensor)
        should_stop = common_util.GlobalDict().get("should_stop", False)
        if should_stop:
            logger.info(
                "should stop is %s from common_util.GlobalDict()" % should_stop
            )
        if step >= self._last_step or should_stop:
            run_context.request_stop()


basic_session_run_hooks.StopAtStepHook.after_run


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
            for key in keys:
                model_fn_results = append_hooks(model_fn_results, key, params)
        return model_fn_results

    Estimator._call_model_fn = dlrover_call_model_fn
