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

from enum import Enum
from typing import List, Optional

from dlrover.python.common.singleton import Singleton
from dlrover.python.training_event.predefined.common import CommonPredefined


class DLRoverCommonEventName(Enum):
    """
    Event in DLRover agent and master
    """

    DLROVER_RENDEZVOUS = "#rendezvous"
    DLROVER_PROCESS_RESTART = "#process_restart"
    DLROVER_PROCESS_FAIL = "#process_fail"
    DLROVER_FAULT_DETECT = "#fault_detect"
    DLROVER_FAULT_TOLERANCE = "#fault_tolerance"
    DLROVER_PRE_CHECK = "#pre_check"


class DLRoverMasterEventName(Enum):
    """
    Event in DLRover master
    """

    DLROVER_MASTER_START = "#master_start"
    DLROVER_MASTER_EXIT = "#master_exit"
    DLROVER_WORKER_CREATE = "#worker_create"
    DLROVER_WORKER_RELAUNCH = "#worker_relaunch"
    DLROVER_WORKER_NO_HEARTBEAT = "#worker_no_heartbeat"
    DLROVER_WORKER_EVENT = "#worker_event"
    DLROVER_JOB_TRAIN = "#job_train"


class DLRoverAgentEventName(Enum):
    """
    Event in DLRover agent
    """

    DLROVER_AGENT_START = "#agent_start"
    DLROVER_AGENT_EXIT = "#agent_exit"
    DLROVER_NETWORK_CHECK = "#network_check"
    DLROVER_ELASTIC_TRAIN = "#elastic_train"


class DLRoverCommonEvent(CommonPredefined, Singleton):
    """Common events for both Master and Agent"""

    def __init__(self, target: str) -> None:
        super().__init__(target)

    def mock(self):
        return self.duration(
            "mock",
            {},
        )

    def rendezvous(
        self,
        rendezvous_type: str,
        round_num: int,
        timeout_sec: int,
        max_nodes: int,
        min_nodes: int,
        **kwargs,
    ):
        """
        Event for rendezvous.

        Master side:
        1. Rendezvous begin: first node join
        2. Rendezvous end: all nodes join

        Agent side:
        1. Rendezvous begin: first node join
        2. Rendezvous end: all nodes join

        Parameters:
        - rendezvous_type: Rendezvous type, such as network-check,
           elastic-train, etc.
        - round_num: Rendezvous round number
        - timeout_sec: Rendezvous timeout in seconds
        - max_nodes: Maximum number of nodes in Rendezvous
        - min_nodes: Minimum number of nodes in Rendezvous
        """
        return self.duration(
            DLRoverCommonEventName.DLROVER_RENDEZVOUS.value,
            {
                "rendezvous_type": rendezvous_type,
                "round_num": round_num,
                "timeout_sec": timeout_sec,
                "max_nodes": max_nodes,
                "min_nodes": min_nodes,
                **kwargs,
            },
        )

    def process_restart(
        self,
        pod_name: str,
        node_name: str,
        restart_round: int,
        errmsg: dict,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_RESTART.value,
            {
                "pod_name": pod_name,
                "node_name": node_name,
                "restart_round": restart_round,
                "errmsg": errmsg,
                **kwargs,
            },
        )

    def process_fail(
        self,
        pod_name: str,
        node_name: str,
        restart_round: int,
        errmsg: dict,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_FAIL.value,
            {
                "pod_name": pod_name,
                "node_name": node_name,
                "restart_round": restart_round,
                "errmsg": errmsg,
                **kwargs,
            },
        )


class DLRoverMasterEvent(DLRoverCommonEvent):
    """DLRover Master events"""

    def __init__(self) -> None:
        super().__init__("dlrover-master")

    def start(self, args: Optional[dict] = None, **kwargs):
        """
        Master start event

        Args:
            args: master start args
            **kwargs:

        Returns:

        """
        if args is None:
            args = {}
        self.instant(
            DLRoverMasterEventName.DLROVER_MASTER_START.value,
            {"params": args, **kwargs},
        )

    def exit(self, exit_code: int, **kwargs):
        """
        Master exit event

        Args:
            exit_code: exit code
            **kwargs:

        Returns:

        """
        return self.instant(
            DLRoverMasterEventName.DLROVER_MASTER_EXIT.value,
            {"exit_code": exit_code, **kwargs},
        )

    def worker_event(
        self,
        pod_name: str,
        from_state: str,
        to_state: str,
        reason: str,
        **kwargs,
    ):
        """Node pod state change event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_WORKER_EVENT.value,
            {
                "pod_name": pod_name,
                "from_state": from_state,
                "to_state": to_state,
                "exit_reason": reason,
                **kwargs,
            },
        )

    def worker_relaunch(self, pod_name: str, relaunch_pod_name: str, **kwargs):
        """Pod relaunch event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_WORKER_RELAUNCH.value,
            {
                "pod_name": pod_name,
                "relaunch_pod_name": relaunch_pod_name,
                **kwargs,
            },
        )

    def worker_create(self, pod_name: str, success: bool, **kwargs):
        """Pod create event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_WORKER_CREATE.value,
            {"pod_name": pod_name, "ok": success, **kwargs},
        )

    def worker_no_heartbeat(self, pod_name: str, timeout: int, **kwargs):
        """Pod create event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_WORKER_NO_HEARTBEAT.value,
            {"pod_name": pod_name, "timeout": timeout, **kwargs},
        )

    def fault_detect(self, reason: str, **kwargs):
        """Fault detection event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_FAULT_DETECT.value,
            {"reason": reason, **kwargs},
        )

    def train_job(self, job_name: str, args: Optional[dict] = None, **kwargs):
        """
        job start event

        Args:
            job_name: job name
            args: job args
            **kwargs: other job params

        Returns:

        """
        return self.duration(
            DLRoverMasterEventName.DLROVER_JOB_TRAIN.value,
            {"job_name": job_name, "params": args, **kwargs},
        )


class DLRoverAgentEvent(DLRoverCommonEvent):
    """DLRover Agent events"""

    def __init__(self) -> None:
        super().__init__("dlrover-agent")

    def network_check(self, round_num: int, node_groups: List[List], **kwargs):
        """Network check event, including network check begin and end.

        Master side:
        1. Network check begin: when network check begin
        2. Network check end: when network check end

        Agent side:
        1. Network check begin: when network check begin
        2. Network check end: when network check end
        """
        return self.duration(
            DLRoverAgentEventName.DLROVER_NETWORK_CHECK.value,
            {"node_groups": node_groups, "round_num": round_num, **kwargs},
        )

    def elastic_train(self, round_num: int, **kwargs):
        """Elastic training event"""
        return self.duration(
            DLRoverAgentEventName.DLROVER_ELASTIC_TRAIN.value,
            {"round_num": round_num, **kwargs},
        )
