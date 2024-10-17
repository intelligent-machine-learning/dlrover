import transformers

from atorch.kernels.patches.patch_llama3_fa3 import llama_fa3_attention_forward
from atorch.utils.import_util import is_flash_attn_3_avaliable
from atorch.utils.version import package_version_smaller_than


def apply_fa3_to_llama3():
    if not is_flash_attn_3_avaliable():
        raise ModuleNotFoundError(f"Please install flash-attention-3")

    if package_version_smaller_than("transformers", "4.43.0"):
        raise NotImplementedError(
            f"transformers version should bigger than 4.43.0, but current version is {transformers.__version__}"
        )
    from transformers.utils import is_flash_attn_2_available

    if not is_flash_attn_2_available():
        raise ModuleNotFoundError(f"flash-attention-2 is needed when using flash-attention-3")

    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_fa3_attention_forward
