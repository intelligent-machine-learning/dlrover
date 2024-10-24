from typing import Dict, List
from dlrover.python.common.time import has_expired
from dlrover.python.diagnosis.common.diagnosis_data import DiagnosisData
from dlrover.python.common.log import default_logger as logger


class DiagnosisDataManager:
    def __init__(self, expire_time_period):
        self.diagnosis_data: Dict[str, List[DiagnosisData]] = {}
        self.expire_time_period = expire_time_period

    def store_data(self, data: DiagnosisData):
        data_type = data.data_type
        if data_type not in self.diagnosis_data:
            logger.debug(f"{data_type} is not found in the store")
            self.diagnosis_data[data_type] = []
        self.diagnosis_data[data_type].append(data)
        self._clean_diagnosis_data(data_type)

    def get_data(self, data_type: str) -> List[DiagnosisData]:
        if data_type not in self.diagnosis_data:
            return []
        return self.diagnosis_data[data_type]

    def _clean_diagnosis_data(self, data_type: str):
        if data_type not in self.diagnosis_data:
            return

        data = self.diagnosis_data[data_type]
        n = 0
        for d in data:
            if has_expired(d.timestamp, self.expire_time_period):
                n = n + 1
            else:
                break

        self.diagnosis_data[data_type] = data[n:]
