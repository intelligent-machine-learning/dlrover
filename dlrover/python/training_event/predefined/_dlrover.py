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

from .common import CommonPredefined


class DLRoverCommonEventName(Enum):
    """Event in DLRover agent and master"""

    DLROVER_START = "#start"
    DLROVER_NODE_JOIN = "#node_join"
    DLROVER_RENDEZVOUS = "#rendezvous"
    DLROVER_NETWORK_CHECK = "#network_check"
    DLROVER_ELASTIC_TRAIN = "#elastic_train"
    DLROVER_PROCESS_RESTART = "#process_restart"
    DLROVER_EXIT = "#exit"


class DLRoverMasterEventName(Enum):
    """Event in DLRover master"""

    DLROVER_POD_CREATE = "#pod_create"
    DLROVER_POD_CHANGE = "#pod_change"
    DLROVER_POD_RELAUNCH = "#pod_relaunch"
    DLROVER_FAULT_DETECT = "#fault_detect"
    DLROVER_REPAIR = "#repair"


class DLRoverCommon(CommonPredefined):
    """Common events for both Master and Agent"""

    def __init__(self, target: str) -> None:
        super().__init__(target)

    def start(self, params: Optional[dict] = None, **kwargs):
        """Event  at DLRover process start"""
        if params is None:
            params = {}
        self.instant(
            DLRoverCommonEventName.DLROVER_START.value,
            {"params": params, **kwargs},
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
        - rendezvous_type: Rendezvous type, such as network-check, elastic-train, etc.
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
            DLRoverCommonEventName.DLROVER_NETWORK_CHECK.value,
            {"node_groups": node_groups, "round_num": round_num, **kwargs},
        )

    def elastic_train(self, round_num: int, **kwargs):
        """Elastic training event"""
        return self.duration(
            DLRoverCommonEventName.DLROVER_ELASTIC_TRAIN.value,
            {"round_num": round_num, **kwargs},
        )

    def node_join(
        self, pod_name: str, node_rank: int, node_name: str, **kwargs
    ):
        """Node join event, including node join and node state change."""
        return self.instant(
            DLRoverCommonEventName.DLROVER_NODE_JOIN.value,
            {
                "pod_name": pod_name,
                "node_rank": node_rank,
                "node_name": node_name,
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

    def exit(self, reason: str, **kwargs):
        """Process exit event, giving exit reason"""
        return self.instant(
            DLRoverCommonEventName.DLROVER_EXIT.value,
            {"reason": reason, **kwargs},
        )


class DLRoverMaster(DLRoverCommon, Singleton):
    """DLRover Master events"""

    def __init__(self) -> None:
        super().__init__("dlrover-master")

    def pod_change(
        self,
        pod_name: str,
        node_name: str,
        from_state: str,
        to_state: str,
        **kwargs,
    ):
        """Node pod state change event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_POD_CHANGE.value,
            {
                "pod_name": pod_name,
                "node_name": node_name,
                "from_state": from_state,
                "to_state": to_state,
                **kwargs,
            },
        )

    def pod_relaunch(
        self, pod_name: str, node_name: str, relaunch_pod_name: str, **kwargs
    ):
        """Pod relaunch event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_POD_RELAUNCH.value,
            {
                "pod_name": pod_name,
                "node_name": node_name,
                "relaunch_pod_name": relaunch_pod_name,
                **kwargs,
            },
        )

    def pod_create(self, pod_name: str, node_rank: int, **kwargs):
        """Pod create event"""
        return self.instant(
            DLRoverMasterEventName.DLROVER_POD_CREATE.value,
            {"pod_name": pod_name, **kwargs},
        )

    def repair(self, reason: str, **kwargs):
        """Self-healing event"""
        return self.duration(
            DLRoverMasterEventName.DLROVER_REPAIR.value,
            {"reason": reason, **kwargs},
        )

    def fault_detect(self, reason: str, **kwargs):
        """Fault detection event"""
        return self.duration(
            DLRoverMasterEventName.DLROVER_FAULT_DETECT.value,
            {"reason": reason, **kwargs},
        )


class DLRoverAgent(DLRoverCommon, Singleton):
    """DLRover Agent events"""

    def __init__(self) -> None:
        super().__init__("dlrover-agent")
