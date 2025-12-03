from pydantic import Field


class NodeResource:
    memory: int  # unit MB
    cpu: float  # The number of cores
    gpu: float
    gpu_type: str
    priority: int


class NodeGroupResource:
    count: int
    resource: NodeResource


class JobResource:
    node_group_resources: dict[str, NodeGroupResource] = Field(default_factory=dict)
    node_resources: dict[str, NodeResource] = Field(default_factory=dict)


class JobMeta:
    uuid: str
    cluster: str
    namespace: str


class JobOptimizePlan:
    timestamp: int
    job_resource: JobResource
    job_meta: JobMeta


class OptimizeConfig:
    optimizer: str
    customized_config: dict[str, str] = Field(default_factory=dict)


