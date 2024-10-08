import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE


def _transpose_weights(moe_mod):
    old_w1_shape = moe_mod.w1.shape
    moe_mod.w1.data.copy_(moe_mod.w1.clone().transpose(-1, -2).reshape(old_w1_shape))
    if hasattr(moe_mod, "v1"):
        old_v1_shape = moe_mod.v1.shape
        moe_mod.v1.data.copy_(moe_mod.v1.clone().transpose(-1, -2).reshape(old_v1_shape))


def transpose_moe(module, target_module_cls=Grouped_GEMM_MoE):
    # moe module and name
    moe_module_and_name = []
    for name, mod_ins in module.named_modules():
        if isinstance(mod_ins, target_module_cls):
            moe_module_and_name.append((name, mod_ins))

    # fsdp name or moe name if no fsdp
    target_fsdp_module_names = {}  # {fsdp_name: [Tuple(moe_name, moe_module)]}
    target_moe_module_names = {}  # {moe_name: moe_module}
    for moe_name, moe_module in moe_module_and_name:
        if "_fsdp_wrapped_module" in moe_name:
            fsdp_index = str(moe_name).rfind("._fsdp_wrapped_module")
            fsdp_name = moe_name[:fsdp_index]
            if fsdp_name not in target_fsdp_module_names:
                target_fsdp_module_names[fsdp_name] = [(moe_name, moe_module)]
            else:
                target_fsdp_module_names[fsdp_name].append((moe_name, moe_module))
        else:
            target_moe_module_names[moe_name] = moe_module

    if len(target_fsdp_module_names) > 0:
        for name, mod_ins in module.named_modules():
            # find fsdp module and transpose moe module instance
            if name in target_fsdp_module_names:
                fsdp_module = mod_ins
                with FullyShardedDataParallel.summon_full_params(fsdp_module, writeback=True), torch.no_grad():
                    for _, moe_mod in target_fsdp_module_names[name]:
                        _transpose_weights(moe_mod)
                fsdp_module._reset_lazy_init()

    for moe_name, moe_mod in target_moe_module_names.items():
        _transpose_weights(moe_mod)

    return module
