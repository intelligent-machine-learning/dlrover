# Copyright 2024 The DLRover Authors. All rights reserved.
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

import threading
import time

from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.diagnosis.common.inference_chain import (
    InferenceAttribute,
    InferenceDescription,
    InferenceName,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import Inference
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.resolver.resolve_training_hang_operator import (  # noqa: E501
    ResolveTrainingHangOperator,
)
from dlrover.python.master.diagnosis.diagnosis import Diagnostician
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)
from dlrover.python.master.diagnosis.precheck_operator import (
    NoPreCheckOperator,
)
from dlrover.python.master.node.job_context import get_job_context

_dlrover_ctx = Context.singleton_instance()


class DiagnosisManager:
    """
    DiagnosisManager is to manage all diagnosis issues in a training job
    """

    def __init__(self):
        self._is_observing_started = False
        self._data_manager: DiagnosisDataManager = DiagnosisDataManager(600)
        self._diagnostician: Diagnostician = Diagnostician(self._data_manager)
        self._job_context = get_job_context()

    @classmethod
    def get_pre_check_operators(cls):
        return [NoPreCheckOperator()]

    def collect_diagnosis_data(self, data: DiagnosisData):
        self._data_manager.store_data(data)

    def pre_check(self):
        if not _dlrover_ctx.pre_check_enable:
            return

        start = time.time()
        pre_check_ops = self.get_pre_check_operators()
        logger.info(
            "Start to training pre-check" f"with operators: {pre_check_ops}."
        )

        for pre_check_op in pre_check_ops:
            current_start = time.time()
            current_op_result = None
            pre_check_op_name = pre_check_op.__class__.__name__

            try:
                # retry loops for each operator
                for i in range(pre_check_op.get_retry_limit_times()):
                    check_start = time.time()

                    # do check
                    current_op_result = pre_check_op.check()
                    logger.info(
                        f"{pre_check_op_name} "
                        f"check({i}) "
                        f"cost: {time.time()-check_start:.2f}ms, "
                        f"result: {current_op_result}"
                    )

                    if not current_op_result.is_success():
                        # try recover and wait
                        pre_check_op.recover()
                        time.sleep(pre_check_op.get_retry_interval_secs())

                        # check again after recover
                        current_op_result = pre_check_op.check()
                    else:
                        break
            except Exception as e:
                logger.error(f"Pre-check operator got unexpected error: {e}")
                continue

            if not current_op_result.is_success():
                action = pre_check_op.get_failed_action()
                self._job_context.enqueue_action(action)
                logger.warning(
                    "Training pre-check failed "
                    f"by {pre_check_op_name} "
                    f"with result: {current_op_result}, "
                    f"cost:{time.time()-current_start:.2f}ms. "
                    f"Invoke action: {action}."
                )
                return
            else:
                logger.info(
                    f"{pre_check_op_name} finish "
                    f"with result: {current_op_result}, "
                    f"cost:{time.time()-current_start:.2f}ms."
                )

        logger.info(
            "Training pre-check complete, " f"cost:{time.time()-start:.2f}ms."
        )

    def start_observing(self):
        logger.info("Start to observing training...")
        self._is_observing_started = True

        self._diagnostician.register_training_problems(
            [
                Inference(
                    InferenceName.TRAINING,
                    InferenceAttribute.ISORNOT,
                    InferenceDescription.HANG,
                )
            ]
        )
        self._diagnostician.register_observers(
            [CheckTrainingHangOperator(self._data_manager)]
        )
        self._diagnostician.register_resolvers(
            [ResolveTrainingHangOperator(self._data_manager)]
        )

        try:
            thread = threading.Thread(
                target=self._diagnose,
                name="diagnose_failures",
                daemon=True,
            )
            thread.start()
            if thread.is_alive():
                logger.info("Diagnosis Manager is started")
        except Exception as e:
            logger.error(
                f"Failed to start the diagnosis manager thread. Error: {e}"
            )

    def stop_observing(self):
        logger.info("Stop Diagnosis Manager to observing training.")
        self._is_observing_started = False

    def _diagnose(self):
        logger.info("Start to diagnose failures for observing.")
        while True:
            if not self._is_observing_started:
                logger.info("Stop to diagnose failures for observing.")
                break

            observed_problems = self._diagnostician.observe_training()
            action = self._diagnostician.resolve_problems(observed_problems)
            self._job_context.enqueue_action(action)

            time.sleep(
                DiagnosisConstant.MASTER_DIAGNOSIS_OBSERVING_INTERVAL_SECS
            )
