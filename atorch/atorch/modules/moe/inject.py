import torch
import torch.distributed as dist

from atorch.modules.moe import get_experts_process_group
from atorch.modules.moe.moe_layer import MoEBertOutput


def boardcast_statedict(state_dict):
    this_rank = dist.get_rank()
    keys = state_dict.keys()  # same order?
    keys = sorted(keys)
    src_rank = 0
    new_state_dict = {}
    with torch.no_grad():
        for key in keys:
            tensor = state_dict[key]
            tensor = tensor.to(torch.cuda.current_device(), copy=True).to(torch.float32)
            if this_rank == src_rank:
                dist.broadcast(tensor, src=src_rank)
                new_state_dict[key] = tensor
            else:
                receiver_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), dtype=torch.float32)
                dist.broadcast(receiver_tensor, src=src_rank)
                new_state_dict[key] = receiver_tensor
    return new_state_dict


def expand_experts_tensor(tensor, num_experts):
    return tensor.expand(num_experts, *tensor.size())


def replace_with_moe(
    layer_obj,
    model,
    config,
    reinitparams=True,
    moe_impl="fastmoe",
    moe_gate="switch",
    num_experts=24,  # how many experts in all world
    outer_batch=True,
    dispatchV2=True,
):
    """
    ```
    class BertLayer:
        def __init__(...):
            self.attention = BertAttention(config)
            self.intermediate = BertIntermediate(config) # replace this two module
            self.output = BertOutput(config)
        def feed_forward_chunk(self, attention_output):# replace this
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output
    ```
    replace_with_moe(BertLayer, model, bert_config, num_experts=num_rank*3)
    """
    if not hasattr(config, "moe_impl"):
        config.moe_impl = moe_impl
    if not hasattr(config, "moe_gate"):
        config.moe_gate = moe_gate
    for _, child in model.named_children():
        if isinstance(child, layer_obj):  # BertLayer

            config.outer_batch = outer_batch
            config.dispatchV2 = dispatchV2
            config.num_experts = num_experts
            output = MoEBertOutput(config)  # random init

            child.feed_forward_chunk = output.forward
            expert_group = get_experts_process_group()  # FIXME: not support copy.deepcopy and torch.save/load
            world_size = expert_group.size() if expert_group else dist.get_world_size()
            num_local_experts = num_experts // world_size
            if not getattr(child, "have_replace_moe", False):
                output.LayerNorm.load_state_dict(child.output.LayerNorm.state_dict())
                it_sd = {
                    key: expand_experts_tensor(value, num_local_experts)
                    for key, value in child.intermediate.dense.state_dict().items()
                }
                output_sd = {
                    key: expand_experts_tensor(value, num_local_experts)
                    for key, value in child.output.dense.state_dict().items()
                }
                if config.moe_impl == "fastmoe":
                    # htoh4 ->weight,bias (num_experts, inter_size, hidden_size)
                    # h4toh -> weight,bias

                    output.mlp.experts.htoh4.load_state_dict(it_sd)
                    output.mlp.experts.h4toh.load_state_dict(output_sd)
                else:
                    # FIXME: atorch moe have no bias
                    output.mlp.experts.inner_experts.data.copy_(it_sd["weight"].transpose(1, 2))
                    output.mlp.experts.out_experts.data.copy_(output_sd["weight"].transpose(1, 2))
                    # restore experts weight from output
                del child.output
                del child.intermediate
                child.moe_output = output
                child.have_replace_moe = True
            # TODO: how to load weight from old dense?
            # setattr(model, name, output)
        else:
            replace_with_moe(
                layer_obj,
                child,
                config,
                moe_impl=moe_impl,
                moe_gate=moe_gate,
                num_experts=num_experts,
            )
