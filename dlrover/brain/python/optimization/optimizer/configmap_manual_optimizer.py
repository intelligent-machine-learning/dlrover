import time
import os

from typing import Optional, Dict, Any
from dlrover.python.common.log import default_logger as logger
from dlrover.brain.python.common.job import (
    JobOptimizePlan,
    JobMeta,
    JobResource,
    NodeGroupResource,
    OptimizeConfig,
    NodeResource,
)
from dlrover.brain.python.common.constants import (
    Node,
)
from dlrover.brain.python.common.constants import (
    UnitConvertor,
)
from dlrover.brain.python.platform.k8s.configmap import ConfigMapReader
from dlrover.brain.python.platform.k8s.monitor import ResourceMonitor

_DEFAULT_MANUAL_SCALE_CONFIGMAP_NAME = "brain-manual-scale-config"
_ENV_MANUAL_SCALE_CONFIGMAP_NAME_KEY = "MANUAL_SCALE_CONFIGMAP_NAME"

class ConfigMapManualOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def get_name() -> str:
        return "ConfigMapManualOptimizer"

    def optimize(self, job: JobMeta, conf: OptimizeConfig) -> JobOptimizePlan:
        conf.customized_config["algorithm"] = "ManualOptimizeJobResource"
        current_node_resource = self.generate_node_resource(job, conf)
        return JobOptimizePlan(
            time=int(time.time() * 1000),
            job_resource=JobResource(
                node_group_resources={
                    Node.NODE_TYPE_WORKER: NodeGroupResource(
                        resource=current_node_resource,
                    )
                },
            ),
            job_meta=job,
        )

    def generate_node_resource(self, job: JobMeta, conf: OptimizeConfig) -> NodeResource:
            manual_scale_configmap_name = os.getenv(_ENV_MANUAL_SCALE_CONFIGMAP_NAME_KEY, _DEFAULT_MANUAL_SCALE_CONFIGMAP_NAME)
            configmap_reader = ConfigMapReader(job.namespace, manual_scale_configmap_name)
            json_data =configmap_reader.read_json_data()
            
            if not isinstance(json_data, dict):
                logger.warning(f"ConfigMap data is not a dictionary, type: {type(json_data)}. Using default empty resource.")
                return NodeResource()
            
            customize_worker_resource: Optional[Dict[str, Any]] = json_data.get("customize_worker_resource", None)
            
            if not customize_worker_resource or not isinstance(customize_worker_resource, dict):
                logger.warning("ConfigMap 'customize_worker_resource' is missing or invalid. Using default empty resource.")
                return NodeResource()
    
            logger.info(f"ConfigMap customize_worker_resource data: {customize_worker_resource}")
    
            cpu_val = customize_worker_resource.get("cpu")
            memory_val = customize_worker_resource.get("memory")
            gpu_nums_val = customize_worker_resource.get("gpus")
            gpu_type_val = customize_worker_resource.get("gpu_type")
    
            required_fields = {"cpu": cpu_val, "memory": memory_val, "gpus": gpu_nums_val, "gpu_type": gpu_type_val}
            missing_fields = [k for k, v in required_fields.items() if v is None]
    
            if missing_fields:
                logger.error(f"Missing required fields in customize_worker_resource: {missing_fields}. Using default empty resource.")
                return NodeResource()

            try:
                gpu_type = str(gpu_type_val).strip()
                if not gpu_type:
                    logger.warning("GPU type is empty, defaulting to 'none' or handling as per business logic.")
                    gpu_type = "none"
                cluster_idle_gpus = ResourceMonitor(job.namespace, gpu_type).get_cluster_idle_gpus()
            except Exception as e:
                logger.warning(f"Cluster check failed: {e}. Fallback to current.")
                return NodeResource()
    
            if not cluster_idle_gpus and cluster_idle_gpus < gpu_nums_val:
                logger.warning(
                    f"No idle GPUs or idle GPUs not enough. Scale up aborted." 
                    f"Need GPUs is {gpu_nums_val}, cluster idle GPUs is {cluster_idle_gpus}."
                )
                return NodeResource()
            
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
                return NodeResource()
            logger.info(
                    f"Applying custom resources -> CPU: {cpu}, Memory: {memory_gib}GiB, GPUs: {gpu_nums}, Type: {gpu_type}"
                )
            return NodeResource(
                    cpu=cpu,
                    memory=memory_bytes,  # GiB → B
                    gpu=gpu_nums,
                    gpu_type=gpu_type,
                )