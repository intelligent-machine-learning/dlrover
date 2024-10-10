from typing import List
from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
)
from dlrover.python.diagnosis.common.constants import (
    DiagnosisAction as DiagnosisActionConstant,
)
from dlrover.python.diagnosis.common.diagnose_action import DiagnoseAction


def coordinate_inferences(inferences: List[Inference]) -> DiagnoseAction:
    action = DiagnoseAction()
    action.add_action(DiagnosisActionConstant.NO_ACTION)
    return action
