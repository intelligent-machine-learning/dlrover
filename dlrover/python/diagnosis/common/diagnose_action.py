from typing import List


class DiagnoseAction:
    def __init__(self):
        self.actions: List[str] = []

    def add_action(self, action: str):
        self.actions.append(action)
