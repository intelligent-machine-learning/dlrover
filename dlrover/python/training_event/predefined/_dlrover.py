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
from typing import Optional

from dlrover.python.common.singleton import Singleton
from dlrover.python.training_event.predefined.common import CommonPredefined


class DLRoverCommonEventName(Enum):
    """
    Event in DLRover agent and master
    """

    DLROVER_RENDEZVOUS = "#rendezvous"
    DLROVER_PROCESS_RESTART = "#process_restart"
    DLROVER_PROCESS_RESTART_MEMBERSHIP = "#process_restart_membership"
    DLROVER_PROCESS_SUCCEEDED = "#process_succeeded"
    DLROVER_PROCESS_FAIL = "#process_fail"
    DLROVER_FAULT_DETECT = "#fault_detect"
    DLROVER_FAULT_TOLERANCE = "#fault_tolerance"
    DLROVER_PRE_CHECK = "#pre_check"
    DLROVER_NETWORK_CHECK = "#network_check"
    DLROVER_JOB_ABORTION = "#job_abortion"
    DLROVER_JOB_RESTART = "#job_restart"


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
    DLROVER_NODE_JOIN = "#node_join"


class DLRoverAgentEventName(Enum):
    """
    Event in DLRover agent
    """

    DLROVER_AGENT_START = "#agent_start"
    DLROVER_AGENT_EXIT = "#agent_exit"


class DLRoverCommonEvent(CommonPredefined, Singleton):
    """Common events for both Master and Agent"""

    def __init__(self, target: str) -> None:
        super().__init__(target)


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

    def node_join(
        self,
        node_id: str,
        node_rank: str,
        node_ip: str,
        rdzv_round: str,
        **kwargs,
    ):
        return self.instant(
            DLRoverMasterEventName.DLROVER_NODE_JOIN.value,
            {
                "node_id": node_id,
                "node_rank": node_rank,
                "node_ip": node_ip,
                "rdzv_round": rdzv_round,
                **kwargs,
            },
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

    def network_check(
        self,
        round: int,
        **kwargs,
    ):
        """
        Network check event
        """
        return self.duration(
            DLRoverCommonEventName.DLROVER_NETWORK_CHECK.value,
            {
                "round": round,
                **kwargs,
            },
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

    def process_restart(
        self,
        pod_name: str,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_RESTART.value,
            {
                "pod_name": pod_name,
                **kwargs,
            },
        )


class DLRoverAgentEvent(DLRoverCommonEvent):
    """DLRover Agent events"""

    def __init__(self) -> None:
        super().__init__("dlrover-agent")

    def start(self, args: Optional[dict] = None, **kwargs):
        """
        Agent start event

        Args:
            args: agent start args
            **kwargs:

        Returns:

        """
        if args is None:
            args = {}
        self.instant(
            DLRoverAgentEventName.DLROVER_AGENT_START.value,
            {"params": args, **kwargs},
        )

    def exit(self, success: bool, **kwargs):
        """
        Master exit event

        Args:
            success: agent exit successfully or not
            **kwargs:

        Returns:

        """
        return self.instant(
            DLRoverAgentEventName.DLROVER_AGENT_EXIT.value,
            {"success": success, **kwargs},
        )

    def rendezvous(
        self,
        rendezvous_type: str,
        node_name: str,
        node_rank: int,
        timeout: int,
        **kwargs,
    ):
        """
        Event for agent rendezvous.
        """
        return self.duration(
            DLRoverCommonEventName.DLROVER_RENDEZVOUS.value,
            {
                "rendezvous_type": rendezvous_type,
                "node_name": node_name,
                "node_rank": node_rank,
                "timeout": timeout,
                **kwargs,
            },
        )

    def network_check(
        self,
        round: int,
        node_rank: int,
        **kwargs,
    ):
        """
        Network check event
        """
        return self.duration(
            DLRoverCommonEventName.DLROVER_NETWORK_CHECK.value,
            {
                "round": round,
                "node_rank": node_rank,
                **kwargs,
            },
        )

    def process_succeeded(
        self,
        node_rank: int,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_SUCCEEDED.value,
            {
                "node_rank": node_rank,
                **kwargs,
            },
        )

    def process_restart(
        self,
        node_rank: int,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_RESTART.value,
            {
                "node_rank": node_rank,
                **kwargs,
            },
        )

    def process_fail(
        self,
        node_rank: int,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_FAIL.value,
            {
                "node_rank": node_rank,
                **kwargs,
            },
        )

    def process_restart_membership(
        self,
        node_rank: int,
        **kwargs,
    ):
        """Process restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_PROCESS_RESTART_MEMBERSHIP.value,
            {
                "node_rank": node_rank,
                **kwargs,
            },
        )

    def job_abortion(
        self,
        node_rank: int,
        reason: str,
        **kwargs,
    ):
        """Job abortion event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_JOB_ABORTION.value,
            {
                "caused_node_rank": node_rank,
                "reason": reason,
                **kwargs,
            },
        )

    def job_restart(
        self,
        node_rank: int,
        reason: str,
        **kwargs,
    ):
        """Job restart event"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_JOB_RESTART.value,
            {
                "caused_node_rank": node_rank,
                "reason": reason,
                **kwargs,
            },
        )
