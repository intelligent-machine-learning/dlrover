from collections import namedtuple

from dlrover.python.common.constants import NodeStatus

NodeStateFlow = namedtuple(
    "NodeStateFlow",
    ("from_status", "to_status", "event_type", "phase", "should_relaunch"),
)

"""
The DAG for the state machine is in the issue
https://github.com/sql-machine-learning/elasticdl/issues/2395#issue-753964852
"""
NODE_STATE_FLOWS = [
    NodeStateFlow(
        from_status=NodeStatus.INITIAL,
        to_status=NodeStatus.PENDING,
        event_type=["ADDED", "MODIFIED"],
        phase="Pending",
        should_relaunch=False,
    ),
    NodeStateFlow(
        from_status=NodeStatus.INITIAL,
        to_status=NodeStatus.RUNNING,
        event_type=["ADDED", "MODIFIED"],
        phase="Running",
        should_relaunch=False,
    ),
    NodeStateFlow(
        from_status=NodeStatus.PENDING,
        to_status=NodeStatus.RUNNING,
        event_type=["ADDED", "MODIFIED"],
        phase="Running",
        should_relaunch=False,
    ),
    NodeStateFlow(
        from_status=NodeStatus.PENDING,
        to_status=NodeStatus.SUCCEEDED,
        event_type=["ADDED", "MODIFIED"],
        phase="Succeeded",
        should_relaunch=False,
    ),
    NodeStateFlow(
        from_status=NodeStatus.PENDING,
        to_status=NodeStatus.FAILED,
        event_type=["ADDED", "MODIFIED"],
        phase="Failed",
        should_relaunch=True,
    ),
    NodeStateFlow(
        from_status=NodeStatus.RUNNING,
        to_status=NodeStatus.SUCCEEDED,
        event_type=["ADDED", "MODIFIED"],
        phase="Succeeded",
        should_relaunch=False,
    ),
    NodeStateFlow(
        from_status=NodeStatus.RUNNING,
        to_status=NodeStatus.FAILED,
        event_type=["ADDED", "MODIFIED"],
        phase="Failed",
        should_relaunch=True,
    ),
    NodeStateFlow(
        from_status=NodeStatus.PENDING,
        to_status=NodeStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=True,
    ),
    NodeStateFlow(
        from_status=NodeStatus.RUNNING,
        to_status=NodeStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=True,
    ),
    NodeStateFlow(
        from_status=NodeStatus.SUCCEEDED,
        to_status=NodeStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=False,
    ),
    NodeStateFlow(
        from_status=NodeStatus.FAILED,
        to_status=NodeStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=False,
    ),
]


def get_node_state_flow(from_status, event_type, phase):
    if event_type == "DELETED" and from_status == NodeStatus.PENDING:
        # The phase if pending if the pending node is deleted.
        phase = NodeStatus.DELETED
    if from_status == phase:
        return None
    for flow in NODE_STATE_FLOWS:
        if (
            from_status == flow.from_status
            and event_type in flow.event_type
            and (flow.phase is None or phase == flow.phase)
        ):
            return flow

    return None
