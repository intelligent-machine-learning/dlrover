import os
import unittest

import deepspeed
import torch
import torch.multiprocessing as mp
from deepspeed import comm as dist
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.module_inject import ReplaceWithTensorSlicing
from deepspeed.ops.op_builder import InferenceBuilder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM

from atorch.rl.ds_hybrid_engine.ds_hook import *  # NOQA
from atorch.rl.ds_hybrid_engine.module_inject.containers import LLAMALayerPolicy
from atorch.rl.ds_hybrid_engine.module_inject.utils import policy_to_ds_container
from atorch.tests.utils.test_utils import init_dist


def _test_llama2_tp(rank, world_size):
    inference_module = InferenceBuilder().load()
    inference_module.allocate_workspace_fp32(1, 1, 1, 2, 30, 30, False, 1, 1, 1)

    init_dist(rank, world_size)
    deepspeed.init_distributed("nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    mp_group_id = global_rank // world_size
    num_mp_groups = world_size // world_size
    for mp_group_id in range(num_mp_groups):
        ranks = list(range(mp_group_id * world_size, (mp_group_id + 1) * world_size, 1))
        mp_group = dist.new_group(ranks)

    torch.manual_seed(1)

    rank = int(os.environ.get("RANK", 0))

    mp_replace = ReplaceWithTensorSlicing(
        mp_group=mp_group, mp_size=world_size, out_dim=0, in_dim=1  # process group for tp inferencing
    )

    # create inference config for Hybrid Engine
    inference_config = DeepSpeedInferenceConfig(
        set_empty_params=True,
        dtype=torch.bfloat16,
        max_out_tokens=40,
        min_out_tokens=60,
        transposed_mode=True,
    )

    model_config = LlamaConfig()

    model_config.num_hidden_layers = 1
    model_config.num_key_value_heads = 64
    model_config.num_attention_heads = 64
    module = LlamaDecoderLayer(model_config)

    module.bfloat16().to(rank)

    # create inference config for Hybrid Engine
    inference_config = DeepSpeedInferenceConfig(
        set_empty_params=True,
        dtype=torch.bfloat16,
        max_out_tokens=40,
        min_out_tokens=60,
        transposed_mode=True,
    )

    # mock tp size = 1
    mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group, mp_size=world_size, out_dim=0, in_dim=1)

    layer_id = 0

    policy_cls = LLAMALayerPolicy

    policy = policy_cls(module, inference=True)

    _container = policy_to_ds_container(
        policy=policy, config=inference_config, model_config=model_config, layer_id=layer_id, child=module
    )

    # mock create inference module in hybrid engine
    _container.set_tensor_parallel_config(world_size, mp_group)
    _container.initialize_tensors(enable_training=True)
    _container.create_ds_model_config()
    _container.create_module()
    _container.set_params_wo_copy(Z3_enabled=True)

    _container.apply_tensor_parallelism(mp_replace, reversed_dim=True)
    _container.module.dtype = torch.bfloat16

    policy.client_module.mlp.up_proj.weighthape_weight_start = (
        policy.client_module.mlp.up_proj.weight.shape[0] // mp_replace.mp_size * (rank + 0)
    )
    shape_weight_end = policy.client_module.mlp.up_proj.weight.shape[0] // mp_replace.mp_size * (rank + 1)

    shape_weight_start = policy.client_module.mlp.up_proj.weight.shape[0] // mp_replace.mp_size * (rank + 0)
    shape_weight_end = policy.client_module.mlp.up_proj.weight.shape[0] // mp_replace.mp_size * (rank + 1)

    assert torch.allclose(
        _container.module.mlp.inter_up_w,
        policy.client_module.mlp.up_proj.weight[shape_weight_start:shape_weight_end, :],
        rtol=1e-05,
        atol=1e-08,
    )

    shape_weight_start = policy.client_module.mlp.gate_proj.weight.shape[0] // mp_replace.mp_size * (rank + 0)
    shape_weight_end = policy.client_module.mlp.gate_proj.weight.shape[0] // mp_replace.mp_size * (rank + 1)

    assert torch.allclose(
        _container.module.mlp.inter_gate_w,
        policy.client_module.mlp.gate_proj.weight[shape_weight_start:shape_weight_end, :],
        rtol=1e-05,
        atol=1e-08,
    )

    shape_weight_start = policy.client_module.mlp.down_proj.weight.shape[1] // mp_replace.mp_size * (rank + 0)
    shape_weight_end = policy.client_module.mlp.down_proj.weight.shape[1] // mp_replace.mp_size * (rank + 1)

    assert torch.allclose(
        _container.module.mlp.output_w,
        policy.client_module.mlp.down_proj.weight[:, shape_weight_start:shape_weight_end],
        rtol=1e-05,
        atol=1e-08,
    )
    assert torch.allclose(_container._4hh_w, policy.client_module.mlp.down_proj.weight, rtol=1e-05, atol=1e-08)

    shape_weight_start = policy.client_module.self_attn.q_proj.weight.shape[0] // mp_replace.mp_size * (rank + 0)
    shape_weight_end = policy.client_module.self_attn.q_proj.weight.shape[0] // mp_replace.mp_size * (rank + 1)

    assert torch.allclose(
        _container.module.attention.attn_qw,
        policy.client_module.self_attn.q_proj.weight[shape_weight_start:shape_weight_end, :],
        rtol=1e-05,
        atol=1e-08,
    )

    shape_weight_start = policy.client_module.self_attn.k_proj.weight.shape[0] // mp_replace.mp_size * (rank + 0)
    shape_weight_end = policy.client_module.self_attn.k_proj.weight.shape[0] // mp_replace.mp_size * (rank + 1)

    assert torch.allclose(
        _container.module.attention.attn_kw,
        policy.client_module.self_attn.k_proj.weight[shape_weight_start:shape_weight_end, :],
        rtol=1e-05,
        atol=1e-08,
    )
    assert torch.allclose(
        _container.module.attention.attn_vw,
        policy.client_module.self_attn.v_proj.weight[shape_weight_start:shape_weight_end, :],
        rtol=1e-05,
        atol=1e-08,
    )

    shape_weight_start = policy.client_module.self_attn.o_proj.weight.shape[1] // mp_replace.mp_size * (rank + 0)
    shape_weight_end = policy.client_module.self_attn.o_proj.weight.shape[1] // mp_replace.mp_size * (rank + 1)

    assert torch.allclose(
        _container.module.attention.attn_ow,
        policy.client_module.self_attn.o_proj.weight[:, shape_weight_start:shape_weight_end],
        rtol=1e-05,
        atol=1e-08,
    )

    assert torch.allclose(
        _container.module.mlp.attn_nw, policy.client_module.input_layernorm.weight, rtol=1e-05, atol=1e-08
    )
    assert torch.allclose(
        _container.module.norm_w, policy.client_module.post_attention_layernorm.weight, rtol=1e-05, atol=1e-08
    )

    batch_size = 4
    seq_len = 16
    hidden_states = 4096

    inputs = torch.ones((batch_size, seq_len, hidden_states)).cuda().bfloat16()
    attention_mask = torch.ones((4, 1, 16, 16)).to(rank)
    """
    hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,


            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,

    """

    _ = policy.client_module.mlp(inputs.bfloat16().to(rank))

    _container.module.attention(inputs.bfloat16().to(rank), attention_mask)
    _ = _container.module.mlp(inputs.bfloat16().to(rank), None, None, None)

    _ = policy.client_module.self_attn(
        inputs.bfloat16().to(rank),
        attention_mask=attention_mask.to(rank),
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )
    _, _, _, _, _ = _container.module.attention(inputs.bfloat16().to(rank), attention_mask)

    _ = policy.client_module(
        inputs.bfloat16().to(rank),
        attention_mask=attention_mask.to(rank),
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )
    hidden_states, _, _ = _container.module(
        inputs.bfloat16().to(rank),
        attention_mask=attention_mask.to(rank),
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )

    model = LlamaForCausalLM(model_config)
    model.bfloat16().to(rank)
    """
    tokenizer = AutoTokenizer.from_pretrained("/mnt1/xuantai.hxd/test/Llama-2-7b-hf")
    inputs = tokenizer("介绍下支付宝?", return_tensors="pt")
    _ = model.generate(inputs.input_ids.to(rank), use_cache=True, max_new_tokens=50)
    """
    decoder_layers = model.model.layers
    layers_length = len(decoder_layers)

    ds_layers = []
    containers = []
    layer_id = 0

    for i in range(layers_length):
        module = decoder_layers[i]
        layer_id += 1
        policy_cls = LLAMALayerPolicy
        policy = policy_cls(module, inference=True)
        _container = policy_to_ds_container(
            policy=policy, config=inference_config, model_config=model_config, layer_id=layer_id, child=module
        )
        # mocker create inference module in hybrid engine
        _container.set_tensor_parallel_config(world_size, mp_group)
        _container.initialize_tensors(enable_training=True)
        _container.create_ds_model_config()
        _container.create_module()
        _container.set_params_wo_copy(Z3_enabled=True)
        _container.apply_tensor_parallelism(mp_replace, reversed_dim=True)
        _container.module.eval()
        _container.module.dtype = torch.bfloat16
        containers.append(_container)
        ds_layers.append(_container.module)

    ds_decoder_layers = torch.nn.ModuleList(ds_layers)

    model.model.layers = ds_decoder_layers
    """
    _ = model.generate(inputs.input_ids.to(rank), use_cache=True, max_new_tokens=50)
    """


@unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
class TestLoadDSModel(unittest.TestCase):
    def test_load_ds_model_with_zero3_partition(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _test_llama2_tp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
