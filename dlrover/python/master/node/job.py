from dlrover.python.diagnosis.common.diagnose_action import DiagnoseActionQueue


class JobContext:
    def __init__(self):
        self._action_queue = DiagnoseActionQueue()

    def enqueue_actions(self, actions):
        for action in actions:
            self._action_queue.add_action(action)

    def next_actions(self, rank):
        return self._action_queue.next_actions(rank)

    def update_context(self):
        pass
