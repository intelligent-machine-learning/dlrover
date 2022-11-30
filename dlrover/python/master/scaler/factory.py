from dlrover.python.master.scaler.elasticjob_scaler import ElasticJobScaler
from dlrover.python.common.constants import EngineType


def new_job_scaler(engine, job_name, namespace):
    if engine == EngineType.KUBERNETES:
        return ElasticJobScaler(job_name, namespace)
