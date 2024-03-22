# copy from deepspeed https://github.com/microsoft/DeepSpeed/
from .module_inject.containers import LLAMALayerPolicy

replace_policies = [
    LLAMALayerPolicy,
]

# non-transformer-based policies
generic_policies = []  # type: ignore
