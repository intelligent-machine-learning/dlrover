# Here we accommodate configs for local sgd/reducer
from dataclasses import dataclass, field
from typing import Optional, Type

from torch.optim import Optimizer
from typing_extensions import Literal


# HACK Add Local SGD related configs
@dataclass
class LocalSGDConfigs:
    # If set use_async=False, normal Local sgd will be used and ranks are synced with steps;
    # If set use_async=True, async Local sgd will be used and ranks are synced with time.
    use_async: bool = False
    # Normal Local SGD
    local_sgd_sync_interval: int = 1
    # Async Local SGD
    local_sgd_sync_time: float = 600  # seconds
    min_total_global_steps: int = 100
    use_step_weight: bool = False
    step_weight_ratio: float = 0.5
    # General parameters
    local_sgd_warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    clip_pseudo_grad: Optional[float] = None
    pseudo_gradnorm_reduce: bool = False
    weight_softmax_temperature: Optional[float] = None
    # anomaly detection related
    skip_anomaly: bool = False
    ewma_alpha: float = 0.02
    ewma_warmup_steps: int = 120
    ewma_threshold: int = 3
    cpu_offload: bool = True


# HACK Add Outer Optimizer related configs
@dataclass
class OuterOptimizerConfigs:
    outer_optim_class: Optional[Type[Optimizer]] = None
    outer_optim_kwargs: dict = field(default_factory=dict)


# HACK Add GTA related configs
@dataclass
class GTAConfigs:
    reducer: Optional[Literal["linear", "gta"]] = None
    consensus_method: Optional[Literal["sum", "count"]] = None
    sparsification_method: Optional[Literal["magnitude", "random", "rescaled_random"]] = None
    normalize: bool = True
    density: float = 1.0
    int8_mask: bool = False


@dataclass
class AnomalyConfigs:
    skip_anomaly: bool = False
    ewma_alpha: float = 0.02
    ewma_warmup_steps: int = 120
    ewma_threshold: int = 3
