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
import importlib
import json
from datetime import datetime
from typing import Dict

from dlrover.python.common.constants import EventReportConstants, NodeStatus
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.common.singleton import Singleton
from dlrover.python.master.elastic_training.net_topology import (
    NodeTopologyMeta,
)
from dlrover.python.scheduler.job import JobArgs
from dlrover.python.training_event import DLRoverMasterEvent
from dlrover.python.training_event.emitter import DurationSpan
from dlrover.python.util.function_util import ignore_exceptions

_master_evt = DLRoverMasterEvent().singleton_instance()


class EventReporter(Singleton):
    def initialize(self, *args, **kwargs):
        """Subclassed can override this method to initialize the reporter."""
        pass

    def is_initialized(self):
        """
        Subclassed can override this method to knnow whether
        the reporter is initialized.
        """
        return True

    @classmethod
    @ignore_exceptions()
    def report(
        cls,
        event_type: str,
        instance: str,
        action: str,
        msg: str,
        labels: Dict[str, str],
    ):
        """
        The basic 'report' implementation, can be overridden by subclasses.

        The default implementation is the outputs logs in the following format:
        [${datetime}][${level}][${instance}][${action}][${msg}][${labels}]
        """

        time_str = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        labels_str = "{}"
        if labels and len(labels) > 0:
            labels_str = json.dumps(labels)

        logger.info(
            f"[{time_str}][{event_type}][{instance}][{action}][{msg}][{labels_str}]"
        )

    # ================ Master Start ================

    @ignore_exceptions()
    def report_master_start(self, args: JobArgs):
        _master_evt.start(args=vars(args))

        self.report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_MASTER_START,
            f"{args.job_name}",
            {},
        )

    @ignore_exceptions()
    def report_master_end(self, args: JobArgs, exit_code: int):
        _master_evt.exit(exit_code=exit_code)

        self.report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_MASTER_END,
            f"{args.job_name}",
            {"exit reason": f"{exit_code}"},
        )

    @ignore_exceptions()
    def report_job_start(self, job_evt: DurationSpan, args: JobArgs):
        job_evt.begin()

        self.report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_JOB_START,
            f"{args.job_name}",
            {},
        )

    @ignore_exceptions()
    def report_job_success(self, job_evt: DurationSpan, args: JobArgs):
        job_evt.success()

        self.report(
            EventReportConstants.TYPE_INFO,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_JOB_SUCCESS,
            f"{args.job_name}",
            {},
        )

    @ignore_exceptions()
    def report_job_fail(
        self, job_evt: DurationSpan, args: JobArgs, error: str
    ):
        job_evt.fail(error=error)

        self.report(
            EventReportConstants.TYPE_ERROR,
            EventReportConstants.JOB_INSTANCE,
            EventReportConstants.ACTION_JOB_FAIL,
            f"{args.job_name}",
            {},
        )

    # ================ Master End ================

    # ================ JobManager Start ================

    @ignore_exceptions()
    def report_node_status_change(
        self, node: Node, old_status: str, new_status: str
    ):
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

        self.report(
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

    @ignore_exceptions()
    def report_process_relaunch(self, node: Node, err_msg=None):
        """Report process relaunching."""

        _master_evt.process_restart(
            pod_name=f"{node.name}",
            errmsg=f"{err_msg}" if err_msg else "",
        )

        self.report(
            event_type=EventReportConstants.TYPE_WARN,
            instance=node.name,
            action=EventReportConstants.ACTION_RESTART_TRAINING,
            msg=f"{node.id}",
            labels={
                "node": node.id,
            },
        )

    @ignore_exceptions()
    def report_node_relaunch(self, node: Node, new_node: Node):
        """Report node relaunching."""

        _master_evt.worker_relaunch(
            pod_name=f"{node.name}",
            relaunch_pod_name=f"{new_node.name}",
            rank=f"{node.rank_index}",
            relaunch_count=f"{node.relaunch_count}",
            max_relaunch_count=f"{node.max_relaunch_count}",
        )

        self.report(
            event_type=EventReportConstants.TYPE_WARN,
            instance=node.name,
            action=EventReportConstants.ACTION_RELAUNCH,
            msg=f"{new_node.id}",
            labels={
                "node": node.host_name,
                "new_node": f"{new_node.name}",
            },
        )

    @ignore_exceptions()
    def report_node_no_heartbeat(self, node: Node, timeout: int):
        """Report node if heartbeat timeout."""

        _master_evt.worker_no_heartbeat(
            pod_name=node.name,
            timeout=timeout,
        )

        self.report(
            event_type=EventReportConstants.TYPE_WARN,
            instance=node.name,
            action=EventReportConstants.ACTION_WORKER_NO_HEARTBEAT,
            msg="",
            labels={},
        )

    # ================ JobManager End ================

    # ================ RDZV Start ================

    @ignore_exceptions()
    def report_rdzv_node_join(
        self,
        node_meta: NodeTopologyMeta,
        rdzv_evt: DurationSpan,
        rdzv_type,
        rdzv_round,
        rdzv_params,
        **kwargs,
    ):
        """Report node joining rendezvous."""

        waiting_nodes = kwargs.get("waiting_nodes", {})
        node_elapsed_time = kwargs.get("node_elapsed_time", -1)
        rdzv_evt.begin()
        _master_evt.node_join(
            node_id=str(node_meta.node_id),
            node_rank=str(node_meta.node_rank),
            node_ip=str(node_meta.node_ip),
            rdzv_round=str(rdzv_round),
            rdzv_type=rdzv_type,
            waiting_nodes=f"{[waiting_nodes.keys()]}",
        )
        self.report(
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

    @ignore_exceptions()
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
        node_rank = kwargs.get("node_rank", [])
        elapsed_time = kwargs.get("elapsed_time", -1)
        rdzv_evt.success(
            node_group=f"{node_ids}",
            elapsed_time=f"{elapsed_time}",
        )
        self.report(
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
                "node_rank": node_rank,
            },
        )

    @ignore_exceptions()
    def report_rdzv_timeout(
        self,
        rdzv_evt: DurationSpan,
        rdzv_type,
        rdzv_round,
        rdzv_params,
        **kwargs,
    ):
        """Report rendezvous timeout."""

        node_id = kwargs.get("node_id", [])
        node_rank = kwargs.get("node_rank", [])
        node_group = kwargs.get("node_group", [])
        elapsed_time = kwargs.get("elapsed_time", -1)
        error_type = kwargs.get("error_type", "")
        error_message = kwargs.get("error_message", "")
        rdzv_evt.fail(
            EventReportConstants.ACTION_RDZV_TIMEOUT,
            elapsed_time=f"{elapsed_time}",
        )

        self.report(
            EventReportConstants.TYPE_WARN,
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
                "error_type": error_type,
                "error_message": error_message,
                "node_id": f"{node_id}",
                "node_rank": f"{node_rank}",
            },
        )

    # ================ RDZV End ================

    def report_network_check_completed(
        self,
        evt: DurationSpan,
        event_type,
        instance,
        action,
        node_status,
        node_check_times,
        node_groups,
    ):
        """Report network check completed"""
        evt.success(
            node_status=node_status,
            node_check_times=node_check_times,
            node_groups=node_groups,
        )

        self.report(
            event_type=event_type,
            instance=instance,
            action=action,
            msg=node_check_times,
            labels={
                "status": node_status,
                "elapsed_time": node_check_times,
            },
        )


context = Context.singleton_instance()


def get_event_reporter(*args, **kwargs) -> EventReporter:
    reporter_cls = context.reporter_cls

    module_name = reporter_cls[0]
    class_name = reporter_cls[1]
    module = importlib.import_module(module_name)

    if hasattr(module, class_name):
        cls = getattr(module, class_name)
        logger.debug(f"Got event reporter: {cls}")
        instance = cls.singleton_instance()
    else:
        logger.debug("Got event reporter: EventReporter")
        instance = EventReporter.singleton_instance()

    if not instance.is_initialized():
        instance.initialize(*args, **kwargs)

    return instance
