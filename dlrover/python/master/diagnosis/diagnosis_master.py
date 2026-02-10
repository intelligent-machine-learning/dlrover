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
from datetime import datetime
from typing import Optional, Dict, Any

from dlrover.python.common.constants import (
    Accelerators,
    EventReportConstants,
    GpuMetricEnum,
    NodeType,
    NpuMetricEnum,
    MthreadsGPUMetricEnum,
    PreCheckStatus,
)
from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.event.reporter import get_event_reporter
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisResult,
    JobHangWatermark,
    DiagnosticianType,
)
from dlrover.python.diagnosis.common.diagnosis_action import NodeAction
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.diagnosis.common.diagnosis_manager import DiagnosisManager
from dlrover.python.diagnosis.diagnostician.node_inconsistency import (
    NodeInconsistencyDiagnostician,
)
from dlrover.python.diagnosis.diagnostician.training_hang import (
    TrainingHangDiagnostician,
)
from dlrover.python.master.diagnosis.diagnosis import Diagnostician
from dlrover.python.master.diagnosis.diagnosis_data_manager import (
    DiagnosisDataManager,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.util.function_util import TimeoutException, timeout
from dlrover.python.util.time_util import get_pending_timeout

_metric_context = JobMetricContext.singleton_instance()
_event_context = JobEventContext.singleton_instance()
_dlrover_context = Context.singleton_instance()


# pending timeout + 600s
def get_pre_check_timeout():
    return get_pending_timeout() + 600


class DiagnosisMaster(DiagnosisManager):
    """
    DiagnosisMaster is used to manage all diagnosis issues in a training job.
    """

    def __init__(self, job_args: Optional[JobArgs] = None):
        self._is_observing_started = False
        self._is_observing_paused = False
        self._data_manager: DiagnosisDataManager = DiagnosisDataManager(600)
        self._diagnostician: Diagnostician = Diagnostician(self._data_manager)
        self._job_context = get_job_context()
        self._job_args = job_args
        self._reporter = get_event_reporter()
        self._metric_monitor = None
        self._lock = threading.Lock()

        super().__init__(self._job_context)

    def collect_diagnosis_data(self, data: DiagnosisData):
        self._data_manager.store_data(data)

    def _report_event(self, event_type, instance, action, msg="", labels=None):
        if labels is None:
            labels = {}
        self._reporter.report(event_type, instance, action, msg, labels)

    @timeout(callback_func=get_pre_check_timeout)
    def pre_check(self):
        if not _dlrover_context.pre_check_enabled():
            self._report_event(
                EventReportConstants.TYPE_INFO,
                EventReportConstants.JOB_INSTANCE,
                EventReportConstants.ACTION_PRE_CHECK_DISABLE,
            )
            logger.info(
                "Pre-check operator config is empty, pre-check disabled."
            )
            self._job_context.set_pre_check_status(PreCheckStatus.DISABLED)
            return

        start = time.time()
        if self._job_context.get_pre_check_status() == PreCheckStatus.PASS:
            logger.info("Skip pre-check for the result is pass.")
            return

        pre_check_ops = _dlrover_context.get_pre_check_operators()
        logger.info(
            "Start training pre-check with "
            f"operators: {[op.__class__.__name__ for op in pre_check_ops]} "
            f"under timeout: {get_pre_check_timeout()}s."
        )

        round = 0
        pre_check_finish = False

        # 1. The all configured pre-check operators will be executed 1 by 1.
        # 2. If any operator check failed, the 'failed actions' will be
        # executed, and all the configured pre-check operators will be
        # executed once more after a 'waiting time' specified by the current
        # operator.
        # 3. There is no retry logic on each operator, because all the
        # operators must re-check if any failed action is executed.
        # 4. If any operator check fails and bypass is set to true, the
        # current result will be ignored, and the process will continue.
        # 5. If the there isn't any 'JobAbortion' during procedure, and the
        # pre-check procedure runs for a long time without finishing, which
        # will be considered as a flaw in the operator's execution. A warning
        # log will be triggered due to a timeout, and the result will be
        # marked as "pass."
        while True:
            logger.info(f"Pre-check round: {round}")
            for index, pre_check_op in enumerate(pre_check_ops):
                if self._job_context.is_stopped():
                    logger.info(
                        f"Training pre-check({round}) interrupted, "
                        f"total time cost:{time.time() - start:.2f}s."
                    )
                    return

                if index == len(pre_check_ops) - 1:
                    is_last_op = True
                else:
                    is_last_op = False

                current_start = time.time()
                pre_check_op_name = pre_check_op.__class__.__name__

                try:
                    check_start = time.time()

                    # do check
                    current_op_result = pre_check_op.check(
                        job_args=self._job_args
                    )
                    logger.info(
                        f"{pre_check_op_name}({index}) done checking, "
                        f"cost: {time.time() - check_start:.2f}s, "
                        f"result: {current_op_result}"
                    )

                    if not current_op_result.is_success():
                        # for fail result
                        if _dlrover_context.is_pre_check_operator_bypass(
                            pre_check_op
                        ):
                            if is_last_op:
                                logger.warning(
                                    f"Set last {pre_check_op_name}"
                                    f"({index}) pre-check pass due "
                                    f"to bypass is enabled."
                                )

                                # break the outer loop
                                pre_check_finish = True
                                break
                            else:
                                logger.warning(
                                    f"Set {pre_check_op_name}({index}) "
                                    "pre-check pass due to bypass is enabled, "
                                    "continue next operator."
                                )
                                # continue inner loop
                                continue
                        else:
                            # go failed actions if check not passed
                            actions = pre_check_op.failed_actions(
                                result_msg=current_op_result.result_msg,
                                abnormal_nodes=current_op_result.abnormal_nodes,  # noqa: E501
                            )
                            self._job_context.enqueue_actions(actions)
                            wait_secs = pre_check_op.get_retry_interval_secs()
                            logger.info(
                                f"{pre_check_op_name} execute failed "
                                f"actions: {actions} and wait for {wait_secs}s"
                            )
                            time.sleep(wait_secs)

                            # break inner loop(start pre-check again)
                            break
                    else:
                        # for success result
                        if is_last_op:
                            # last op, break the outer loop
                            logger.info(
                                f"Last operator {pre_check_op_name} passed "
                                f"with result: {current_op_result}, "
                                f"cost:{time.time() - current_start:.2f}s."
                            )
                            pre_check_finish = True
                            break
                        else:
                            # not last op, keep going
                            logger.info(
                                f"Operator {pre_check_op_name} passed "
                                f"with result: {current_op_result}, "
                                f"cost:{time.time() - current_start:.2f}s, "
                                "continue next operator."
                            )
                            continue
                except TimeoutException as te:
                    self._report_event(
                        EventReportConstants.TYPE_WARN,
                        EventReportConstants.JOB_INSTANCE,
                        EventReportConstants.ACTION_PRE_CHECK_TIMEOUT,
                        pre_check_op.__class__.__name__,
                    )
                    raise te
                except Exception as e:
                    self._report_event(
                        EventReportConstants.TYPE_WARN,
                        EventReportConstants.JOB_INSTANCE,
                        EventReportConstants.ACTION_PRE_CHECK_ERROR,
                        pre_check_op.__class__.__name__,
                    )
                    logger.warning(
                        f"{pre_check_op.__class__.__name__} got unexpected error: {e}"
                    )
                    if is_last_op:
                        pre_check_finish = True
                    else:
                        continue

            # outer loop continue here
            if pre_check_finish:
                self._job_context.set_pre_check_status(PreCheckStatus.PASS)
                self._report_event(
                    EventReportConstants.TYPE_INFO,
                    EventReportConstants.JOB_INSTANCE,
                    EventReportConstants.ACTION_PRE_CHECK_PASS,
                )
                break
            else:
                round += 1
                continue

        logger.info(
            f"Training pre-check complete, cost:{time.time() - start:.2f}s."
        )

    def new_metric_monitor(self, monitor):
        self._metric_monitor = monitor

    def start_metric_collect(self):
        """
        create a XpuMetricMonitor instance based on worker XPU type
        start a thread to collect metrics data
        store the data into global JobMetricContext

        """
        logger.info(f"start {self._job_args.xpu_type} metric collector...")
        if self._metric_monitor:
            self._metric_monitor.start()

    def stop_metric_collect(self):
        logger.info("Stop metric collector...")
        if self._metric_monitor:
            self._metric_monitor.stop()

    def join_metric_collect(self):
        logger.info("Join metric collector...")
        if self._metric_monitor:
            self._metric_monitor.join()

    def pause_observing(self):
        if not self._is_observing_paused:
            logger.info("Pause observing training...")
            _event_context.train_steps.clear_step_events()
            _metric_context.clear_node_metrics()
            self._is_observing_paused = True

    def continue_observing(self):
        if self._is_observing_paused:
            logger.info("Continue observing training...")
            _event_context.train_steps.clear_step_events()
            _metric_context.clear_node_metrics()

            self._is_observing_paused = False

    def get_diagnosis_inputs(self) -> Dict[str, Any]:
        return {"job_nodes": self._job_context.job_nodes()}

    def _register_diagnosticians(self):
        self.register_diagnostician(
            DiagnosticianType.NODE_INCONSISTENCY,
            NodeInconsistencyDiagnostician(self._job_args),
            60 * 5,
        )

        self.register_diagnostician(
            DiagnosticianType.TRAINING_HANG,
            TrainingHangDiagnostician(self._data_manager),
            60 * 10,
        )

    def start_observing(self):
        logger.info("Start to observing training...")
        self._is_observing_started = True

        # register periodical diagnosis
        self._register_diagnosticians()
        try:
            self.start_diagnosis()

            diag_metric = threading.Thread(
                target=self._diagnose_metrics,
                name="diagnose_metrics",
                daemon=True,
            )
            diag_metric.start()
            if diag_metric.is_alive():
                logger.info("_diagnose_metrics thread has started")

        except Exception as e:
            logger.error(
                f"Failed to start the diagnosis manager thread. Error: {e}"
            )

    def stop_observing(self):
        logger.info("Stop diagnosis manager training observation...")
        self._is_observing_started = False

    def check_tensor_drop_zero(self, duration):
        if duration > _metric_context.max_metric_records:
            duration = _metric_context.max_metric_records

        if self._job_args.xpu_type is Accelerators.ASCEND_NPU:
            metrics = _metric_context.backtrace_avg_metrics(
                NpuMetricEnum.NPU_UTIL, duration
            )
        elif self._job_args.xpu_type is Accelerators.NVIDIA_GPU:
            metrics = _metric_context.backtrace_avg_metrics(
                GpuMetricEnum.GPU_TENSOR_UTIL, duration
            )
        elif self._job_args.xpu_type is Accelerators.MTHREADS_GPU:
            metrics = _metric_context.backtrace_avg_metrics(
                MthreadsGPUMetricEnum.GPU_TENSOR_UTIL, duration
            )
        else:
            return DiagnosisResult.DIAG_INVALID_PARAM, 0, 0

        if metrics is None:
            logger.warning(f"invalid metrics: {metrics}")
            return DiagnosisResult.DIAG_ERROR, 0, 0

        if len(metrics) < duration:
            logger.debug(
                f"Waiting for tensor metrics: {len(metrics)}/{duration}"
            )
            return DiagnosisResult.DIAG_WAITING, 0, 0

        key_list = list(metrics.keys())
        key_list.sort(reverse=True)

        logger.debug(f"Check tensor metrics: {dict(metrics)}")

        start_ts = key_list[0]
        end_ts = key_list[0]
        for key in key_list:
            end_ts = key
            if metrics[key] > JobHangWatermark.TENSOR_UTIL_LOW_WM:
                return DiagnosisResult.DIAG_HEALTHY, start_ts, end_ts

            duration = duration - 1
            if duration <= 0:
                break

        return DiagnosisResult.DIAG_HANG, start_ts, end_ts

    def _diagnose_metrics(
        self, interval=DiagnosisConstant.METRIC_COLLECT_INTERVAL_SECS
    ):
        hang_downtime = _dlrover_context.hang_downtime

        while True:
            if not self._is_observing_started:
                logger.info(
                    f"Stop _metric_diagnose thread: {self._is_observing_started}"
                )
                break

            if self._is_observing_paused:
                logger.info(
                    f"Pause _metric_diagnose thread: {self._is_observing_paused}"
                )
                time.sleep(DiagnosisConstant.METRIC_COLLECT_INTERVAL_SECS)
                continue

            result, start, end = self.check_tensor_drop_zero(hang_downtime)
            step_hang = _event_context.check_job_step_hang()

            if result is DiagnosisResult.DIAG_HANG:
                start_dt = datetime.fromtimestamp(start).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                end_dt = datetime.fromtimestamp(end).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                logger.warning(
                    f"Detect job hang by tensor drop zero: "
                    f"{start_dt}-{end_dt}, step hang is {step_hang}"
                )

                if _dlrover_context.hang_detection == 2:
                    if step_hang is True:
                        node = self._job_context.job_node_by_rank(
                            NodeType.WORKER, 0
                        )
                        if node is None:
                            logger.warning("Failed to get rank 0 worker")
                        else:
                            logger.info(
                                f"Restart worker-{node.id} all processes"
                            )
                            _event_context.train_steps.clear_step_events()
                            self._job_context.enqueue_diagnosis_action(
                                NodeAction(
                                    node_id=node.id,
                                    node_type=NodeType.WORKER,
                                    action_type=(
                                        DiagnosisActionType.RESTART_WORKER
                                    ),
                                    instance=DiagnosisConstant.ANY_INSTANCE,
                                )
                            )

            time.sleep(interval)
