
from dlrover.brain.python.common.job import (
    JobMeta,
    NodeResource,
    OptimizeConfig,
)
from dlrover.brain.python.common.constants import (
    UnitConvertor,
)
from dlrover.brain.python.platform.k8s.configmap import ConfigMapReader
from dlrover.python.common.log import default_logger as logger
from typing import Optional, Dict, Any

class BaseOptimizeJobResource:
    def __init__(self):
        self.current_node_resource = NodeResource()
        pass

    @staticmethod
    def get_name() -> str:
        return "BaseOptimizeJobResource"

    def generate_node_resource(self, job: JobMeta, conf: OptimizeConfig) -> NodeResource:
        configmap_reader = ConfigMapReader(job.namespace, "brain-base-config")
        json_data =configmap_reader.read_json_Data()
        
        if not isinstance(json_data, dict):
            logger.warning(f"ConfigMap data is not a dictionary, type: {type(json_data)}. Using default empty resource.")
            return self.current_node_resource
        
        customize_worker_resource: Optional[Dict[str, Any]] = json_data.get("customize_worker_resource", None)
        
        if not customize_worker_resource or not isinstance(customize_worker_resource, dict):
            logger.warning("ConfigMap 'customize_worker_resource' is missing or invalid. Using default empty resource.")
            return self.current_node_resource

        logger.info(f"ConfigMap customize_worker_resource data: {customize_worker_resource}")

        cpu_val = customize_worker_resource.get("cpu")
        memory_val = customize_worker_resource.get("memory")
        gpu_nums_val = customize_worker_resource.get("gpus")
        gpu_type_val = customize_worker_resource.get("gpu_type")

        required_fields = {"cpu": cpu_val, "memory": memory_val, "gpus": gpu_nums_val, "gpu_type": gpu_type_val}
        missing_fields = [k for k, v in required_fields.items() if v is None]

        if missing_fields:
            logger.error(f"Missing required fields in customize_worker_resource: {missing_fields}. Using default empty resource.")
            return self.current_node_resource
        
        try:
            cpu = int(cpu_val)
            memory_gib = int(memory_val) # GiB
            gpu_nums = int(gpu_nums_val)
            
            if cpu < 0 or memory_gib < 0 or gpu_nums < 0:
                raise ValueError("Resource values cannot be negative.")
            
            # unit conversion (GiB -> Bytes)
            memory_bytes = memory_gib * UnitConvertor.GIB_TO_BYTES
            gpu_type = str(gpu_type_val).strip()
            if not gpu_type:
                logger.warning("GPU type is empty, defaulting to 'none' or handling as per business logic.")
                gpu_type = "none" 

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data type in customize_worker_resource: {e}. Using default resource.")
            return self.current_node_resource
        logger.info(
                f"Applying custom resources -> CPU: {cpu}, Memory: {memory_gib}GiB, GPUs: {gpu_nums}, Type: {gpu_type}"
            )
        return NodeResource(
                cpu=cpu,
                memory=memory_bytes,  # GiB → B
                gpu=gpu_nums,
                gpu_type=gpu_type,
            )
