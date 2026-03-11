# Copyright 2025 The DLRover Authors. All rights reserved.
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
from datetime import datetime
from typing import Optional

from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.node import Node
from dlrover.python.diagnosis.common.constants import (
    DiagnosisErrorConstant,
    JobHangWatermark,
    DiagnosisResult,
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
import re
import sys
from typing import Dict, List, Tuple
from dlrover.python.diagnosis.common.constants import DiagnosisDataType
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    EventAction,
    NoAction,
    NodeAction,
)
from dlrover.python.common.constants import (
    EventReportConstants,
    Accelerators,
    NpuMetricEnum,
    GpuMetricEnum,
    MthreadsGPUMetricEnum,
    NodeType,
    HangDetectionStrategy,
)
from dlrover.python.master.node.job_context import get_job_context

_dlrover_context = Context.singleton_instance()
_job_context = get_job_context()
_event_context = JobEventContext.singleton_instance()
_metric_context = JobMetricContext.singleton_instance()
HANG_METRIC_PREFIX = "XPU_TIMER_COMMON_HANG"


class TrainingHangDiagnostician(Diagnostician):
    """
    TrainingHangDiagnostician is to observe and resolve the training hang problem
    """

    def __init__(self, job_args, data_mgr):
        super().__init__(job_args)
        self._data_mgr = data_mgr

    def observe(self, **kwargs) -> Optional[DiagnosisObservation]:
        # analyze xpu_timer metrics 1st
        xpu_timer_diagnosis_data = self._data_mgr.get_data(
            DiagnosisDataType.XPU_TIMER_METRIC
        )

        if self.is_hang_by_xpu_timer_metric(xpu_timer_diagnosis_data):
            return DiagnosisObservation(
                observation=DiagnosisErrorConstant.TRAINING_IS_HANG,
                extra_infos={"TYPE": "BY_XPU_TIMER_METRIC"},
            )

        # analyze other metrics
        if self.is_hang_by_other_metric():
            return DiagnosisObservation(
                observation=DiagnosisErrorConstant.TRAINING_IS_HANG,
                extra_infos={"TYPE": "BY_OTHER_METRIC"},
            )

        return None

    def resolve(
        self, problem: DiagnosisObservation, **kwargs
    ) -> List[DiagnosisAction]:
        if (
            problem is not None
            and problem.observation == DiagnosisErrorConstant.TRAINING_IS_HANG
        ):
            actions: List[DiagnosisAction] = []

            # add metric collection action for all worker node
            for _, node in _job_context.job_nodes_by_type(
                NodeType.WORKER
            ).items():
                if not node.is_released:
                    actions.append(
                        NodeAction(
                            node_id=node.id,
                            node_type=NodeType.WORKER,
                            action_type=DiagnosisActionType.COLLECT_METRIC,
                            instance=node.id,
                        )
                    )

            # do failover
            if (
                _dlrover_context.hang_detection
                == HangDetectionStrategy.DO_FAILOVER
            ):
                node_0: Optional[Node] = _job_context.job_node_by_rank(
                    NodeType.WORKER, 0
                )
                if node_0 is None:
                    logger.warning("Failed to get rank 0 worker")
                else:
                    logger.info(f"Restart worker-{node_0.id} all processes")
                    _event_context.train_steps.clear_step_events()

                    actions.append(
                        NodeAction(
                            node_id=node_0.id,
                            node_type=NodeType.WORKER,
                            action_type=(DiagnosisActionType.RESTART_WORKER),
                            instance=DiagnosisConstant.ANY_WORKER_INSTANCE,
                        )
                    )
            # do event notify
            elif (
                _dlrover_context.hang_detection
                == HangDetectionStrategy.DO_NOTIFY
            ):
                actions.append(
                    EventAction(
                        event_type=EventReportConstants.TYPE_WARN,
                        event_instance=EventReportConstants.JOB_INSTANCE,
                        event_action=problem.observation,
                        event_msg="",
                        event_labels={},
                        expired_time_period=120,
                    )
                )
            # log only
            else:
                logger.info(
                    f"Got {DiagnosisErrorConstant.TRAINING_IS_HANG} "
                    "but action is disabled(log only)."
                )
            return actions
        return [NoAction()]

    def is_hang_by_xpu_timer_metric(self, diagnosis_data: List[DiagnosisData]):
        if not diagnosis_data or len(diagnosis_data) <= 0:
            logger.debug("Skip for no worker xpu-timer metric.")
            return False

        logger.debug(
            "Hang detection start using diagnosis data by xpu-timer, "
            f"data number: {len(diagnosis_data)}, "
            f"data size: {sys.getsizeof(diagnosis_data)}."
        )
        worker_hang_metric: Dict[int, List[Tuple[int, bool]]] = {}

        # the format of the hang metric can refer these files:
        # dlrover/python/tests/data/xpu_timer/hang
        for data in diagnosis_data:
            # filter hang metric
            each_worker_metric = [
                line
                for line in data.data_content.splitlines()
                if line.startswith(HANG_METRIC_PREFIX)
            ]

            # if all local rank is hanged, tag worker hang
            rank_hang_size = 0
            is_worker_hang = False
            for each_rank_metric in each_worker_metric:
                match = re.search(r"(\d+)(?!.*\d)", each_rank_metric)
                if match and match.group(0) == "1":
                    rank_hang_size += 1
            if rank_hang_size == len(each_worker_metric):
                is_worker_hang = True

            if data.node_rank not in worker_hang_metric:
                worker_hang_metric[data.node_rank] = []
            worker_hang_metric[data.node_rank].append(
                (data.timestamp, is_worker_hang)
            )

        # hang detection rules:
        # 1. 100% worker got hang metric
        # 2. last for 5+ minutes
        hang_id, hang_last = self._get_hang_overlaps(worker_hang_metric)
        hang_last_threshold = self._get_hang_time_last_threshold()
        if hang_id != -1 and hang_last > hang_last_threshold:
            logger.info(
                f"Got hang worker: {hang_id}, time last: {hang_last}, "
                f"threshold: {hang_last_threshold}"
            )
            return True
        return False

    def is_hang_by_other_metric(self):
        hang_downtime = _dlrover_context.hang_downtime
        is_tensor_drop_zero, start, end = self._check_tensor_drop_zero(
            hang_downtime
        )
        step_hang = _event_context.check_job_step_hang()
        ckpt_hang = _event_context.check_ckpt_hang()
        event_block = _event_context.check_event_block()

        if is_tensor_drop_zero == DiagnosisResult.DIAG_HANG:
            time_format = "%Y-%m-%d %H:%M:%S"
            start_dt = datetime.fromtimestamp(start).strftime(time_format)
            end_dt = datetime.fromtimestamp(end).strftime(time_format)
            logger.warning(
                f"Detect job hang by tensor drop zero: "
                f"{start_dt}-{end_dt}, step hang is {step_hang}, "
                f"ckpt hang is {ckpt_hang}"
            )

            # set to hang when both is_tensor_drop_zero and (step_hang or ckpt_hang)
            if step_hang or ckpt_hang or event_block:
                return True
        return False

    def _check_tensor_drop_zero(self, duration):
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

    def _get_hang_time_last_threshold(self):
        # set 5 minutes for now(second)
        return 5 * 60

    def _get_hang_overlaps(
        self, worker_hang_metric: Dict[int, List[Tuple[int, bool]]]
    ) -> Tuple[int, int]:
        """
        Require all workers hang from latest and find the hang overlaps.

        Args:
            worker_hang_metric (Dict[int, List[Tuple[int, bool]]]): Input
                metric in format: node_id: [(timestamp, is_hang), ...]

        Returns:
            The hang overlaps' id and time last in tuple format.
        """

        worker_hang_length_min = 0
        worker_hang_id = -1

        # find the intersection from latest
        for worker_id, tuple_list in worker_hang_metric.items():
            # sorted by timestamp
            tuple_list.sort(key=lambda x: x[0])
            worker_hang_length = 0

            # ort in descending order
            reversed_tuple_list = reversed(tuple_list)
            for tuple_item in reversed_tuple_list:
                if tuple_item[1]:
                    worker_hang_length += 1
                else:
                    break

            if worker_hang_length > 0:
                if worker_hang_length_min == 0:
                    worker_hang_length_min = worker_hang_length
                    worker_hang_id = worker_id
                elif worker_hang_length < worker_hang_length_min:
                    worker_hang_length_min = worker_hang_length
                    worker_hang_id = worker_id
            else:
                # there is normal worker
                return -1, -1

        # get the intersection's time last
        if worker_hang_id != -1 and worker_hang_length_min != 0:
            hang_worker_metric = worker_hang_metric[worker_hang_id]
            time_last = (
                hang_worker_metric[len(hang_worker_metric) - 1][0]
                - hang_worker_metric[
                    len(hang_worker_metric) - worker_hang_length_min
                ][0]
            )
            return worker_hang_id, time_last

        return -1, -1
