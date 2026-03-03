from pydantic import (
    Field,
    BaseModel,
)


class NodeResource(BaseModel):
    memory: int = 0
    cpu: float = 0.0  # The number of cores
    gpu: float = 0.0
    gpu_type: str = ""
    priority: int = 0


class NodeGroupResource(BaseModel):
    count: int = 0
    resource: NodeResource = NodeResource()


class JobResource(BaseModel):
    node_group_resources: dict[str, NodeGroupResource] = Field(default_factory=dict)
    node_resources: dict[str, NodeResource] = Field(default_factory=dict)


class JobMeta(BaseModel):
    uuid: str = ""
    cluster: str = ""
    namespace: str = ""


class JobOptimizePlan(BaseModel):
    timestamp: int = 0
    job_resource: JobResource = JobResource()
    job_meta: JobMeta = JobMeta()


class OptimizeConfig(BaseModel):
    optimizer: str = ""
    customized_config: dict[str, str] = Field(default_factory=dict)


