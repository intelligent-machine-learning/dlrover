from dlrover.python.master.diagnosis import (
    analyst,
    diagnosis_data,
    diagnostician,
)


class DiagnosisManager:
    def __init__(self):
        self.data_manager = diagnosis_data.DataManager()
        self.analyst = analyst.Analyst()
        self.diagnostician = diagnostician.