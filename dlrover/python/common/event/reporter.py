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

import json
from datetime import datetime
from typing import Dict

from dlrover.python.common.constants import EventReportConstants, NodeStatus
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.training_event import DLRoverMasterEvent
from dlrover.python.training_event.emitter import DurationSpan

_master_evt = DLRoverMasterEvent().singleton_instance()


class EventReporter(Singleton):
    @classmethod
    def inner_report(
        cls,
        event_type: str,
        instance: str,
        action: str,
        msg: str,
        labels: Dict[str, str],
    ):
        time_str = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        labels_str = "{}"
        if labels and len(labels) > 0:
            labels_str = json.dumps(labels)

        logger.info(
            f"[{time_str}][{event_type}][{instance}]"
            f"[{action}][{msg}][{labels_str}]"
        )

    # ================ Master Start ================

    def report_master_start(self, args: JobArgs):
        _master_evt.start(args=vars(args))

        self.inner_report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_MASTER_START,
            f"{args.job_name}",
            {},
        )

    def report_master_end(self, args: JobArgs, exit_code: int):
        _master_evt.exit(exit_code=exit_code)

        self.inner_report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_MASTER_END,
            f"{args.job_name}",
            {"exit reason": f"{exit_code}"},
        )

    def report_job_start(self, job_evt: DurationSpan, args: JobArgs):
        job_evt.begin()

        self.inner_report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_JOB_START,
            f"{args.job_name}",
            {},
        )

    def report_job_success(self, job_evt: DurationSpan, args: JobArgs):
        job_evt.success()

        self.inner_report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_JOB_SUCCESS,
            f"{args.job_name}",
            {},
        )

    def report_job_fail(
        self, job_evt: DurationSpan, args: JobArgs, error: str
    ):
        job_evt.fail(error=error)

        self.inner_report(
            EventReportConstants.TYPE_ERROR,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_JOB_FAIL,
            f"{args.job_name}",
            {},
        )

    # ================ Master End ================

    # ================ JobManager Start ================

    def report_node_status_change(self, node, old_status, new_status):
        """Report node status when changing."""

        _master_evt.worker_event(
            pod_name=node.name,
            from_state=old_status,
            to_state=new_status,
            reason=node.exit_reason,
        )

        event_type = EventReportConstants.TYPE_INFO
        if new_status in [NodeStatus.FAILED, NodeStatus.DELETED]:
            event_type = EventReportConstants.TYPE_WARN

        self.inner_report(
            event_type=event_type,
            instance=node.name,
            action=EventReportConstants.ACTION_STATUS_UPDATE,
            msg=f"{old_status} to {new_status}",
            labels={
                "from_state": old_status,
                "to_state": new_status,
                "node": node.name,
                "exit reason": node.exit_reason,
            },
        )

    def report_node_relaunch(self, node, new_node):
        """Report node when relaunching."""

        _master_evt.worker_relaunch(
            pod_name=node.name,
            relaunch_pod_name=new_node.name,
        )

        self.inner_report(
            event_type=EventReportConstants.TYPE_WARN,
            instance=node.name,
            action=EventReportConstants.ACTION_RELAUNCH,
            msg=f"{new_node.id}",
            labels={
                "node": node.host_name,
                "new_node": f"{new_node.name}",
            },
        )

    def report_node_no_heartbeat(self, node, timeout):
        """Report node if heartbeat timeout."""

        _master_evt.worker_no_heartbeat(
            pod_name=node.name,
            timeout=timeout,
        )

        self.inner_report(
            event_type=EventReportConstants.TYPE_WARN,
            instance=node.name,
            action=EventReportConstants.ACTION_WORKER_NO_HEARTBEAT,
            msg="",
            labels={},
        )

    # ================ JobManager End ================

    # ================ RDZV Start ================

    def report_rdzv_node_join(
        self, node_meta, rdzv_type, rdzv_round, rdzv_params, **kwargs
    ):
        """Report node joining rendezvous."""

        waiting_nodes = kwargs.get("waiting_nodes", {})
        node_elapsed_time = kwargs.get("node_elapsed_time", -1)
        _master_evt.node_join(
            node_id=str(node_meta.node_id),
            node_rank=str(node_meta.node_rank),
            node_ip=str(node_meta.node_ip),
            rdzv_round=str(rdzv_round),
            rdzv_type=rdzv_type,
            waiting_nodes=waiting_nodes.keys(),
        )
        self.inner_report(
            EventReportConstants.TYPE_INFO,
            str(node_meta.node_rank),
            EventReportConstants.ACTION_RDZV_JOIN,
            f"{rdzv_type}={rdzv_round}",
            {
                "rendezvous_type": rdzv_type,
                "max_node": f"{rdzv_params.max_nodes}",
                "min_node": f"{rdzv_params.min_nodes}",
                "waiting_nodes": f"{[waiting_nodes.keys()]}",
                "node_elapsed_time": f"{node_elapsed_time}",
            },
        )

    def report_rdzv_complete(
        self,
        rdzv_evt: DurationSpan,
        rdzv_type,
        rdzv_round,
        rdzv_params,
        **kwargs,
    ):
        """Report rendezvous complete."""

        node_ids = kwargs.get("node_ids", [])
        elapsed_time = kwargs.get("elapsed_time", -1)
        rdzv_evt.success(
            node_group=f"{node_ids}",
            elapsed_time=f"{elapsed_time}",
        )
        self.inner_report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_RDZV_COMPLETE,
            f"{rdzv_type}={rdzv_round}",
            {
                "rendezvous_type": rdzv_type,
                "status": "success",
                "max_nodes": f"{rdzv_params.max_nodes}",
                "min_nodes": f"{rdzv_params.min_nodes}",
                "node_ids": f"{node_ids}",
                "elapsed_time": f"{elapsed_time}",
                "error_message": "",
            },
        )

    def report_rdzv_timeout(
        self,
        rdzv_evt: DurationSpan,
        rdzv_type,
        rdzv_round,
        rdzv_params,
        **kwargs,
    ):
        """Report rendezvous timeout."""

        node_group = kwargs.get("node_group", [])
        elapsed_time = kwargs.get("elapsed_time", -1)
        rdzv_evt.fail(
            EventReportConstants.ACTION_RDZV_TIMEOUT,
            elapsed_time=f"{elapsed_time}",
        )

        self.inner_report(
            EventReportConstants.TYPE_ERROR,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_RDZV_TIMEOUT,
            f"{rdzv_type}={rdzv_round}",
            {
                "rendezvous_type": rdzv_type,
                "status": "timeout",
                "max_nodes": f"{rdzv_params.max_nodes}",
                "min_nodes": f"{rdzv_params.min_nodes}",
                "node_group": json.dumps(node_group),
                "elapsed_time": f"{elapsed_time}",
                "error_message": "",
            },
        )

    # ================ RDZV End ================


def get_event_reporter() -> EventReporter:
    return EventReporter.singleton_instance()
