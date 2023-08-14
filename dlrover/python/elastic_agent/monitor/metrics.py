from dataclasses import dataclass


@dataclass
class GPUMetric:
    index: int
    total_memory_mb: int
    used_memory_mb: int
    gpu_utilization: float
