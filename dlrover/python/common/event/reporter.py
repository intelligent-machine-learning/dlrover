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

from dlrover.python.common.constants import EventReportConstants
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.training_event import DLRoverMasterEvent

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
                "node_group": f"{[waiting_nodes.keys()]}",
                "node_elapsed_time": f"{node_elapsed_time}",
            },
        )

    def report_rdzv_complete(
        self, rdzv_evt, rdzv_type, rdzv_round, rdzv_params, **kwargs
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
                "node_group": f"{node_ids}",
                "elapsed_time": f"{elapsed_time}",
                "error_message": "",
            },
        )

    def report_rdzv_timeout(
        self, rdzv_evt, rdzv_type, rdzv_round, rdzv_params, **kwargs
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
