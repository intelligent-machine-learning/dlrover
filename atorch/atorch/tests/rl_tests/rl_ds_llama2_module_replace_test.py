import unittest

import torch
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.module_inject import ReplaceWithTensorSlicing
from deepspeed.ops.op_builder import InferenceBuilder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM

from atorch.rl.ds_hybrid_engine.ds_hook import *  # NOQA
from atorch.rl.ds_hybrid_engine.module_inject.containers import LLAMALayerPolicy
from atorch.rl.ds_hybrid_engine.module_inject.utils import policy_to_ds_container


@unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
class TestDsLlama2Container(unittest.TestCase):
    def test_ds_llama2_hybrid_engine(self):
        torch.manual_seed(1)
        inference_module = InferenceBuilder().load()
        inference_module.allocate_workspace_fp32(1, 1, 1, 2, 30, 30, False, 1, 1, 1)
        model_config = LlamaConfig()
        model_config.num_hidden_layers = 2
        model_config.num_key_value_heads = 8
        model_config.num_attention_heads = 64
        module = LlamaDecoderLayer(model_config)
        module.bfloat16().to(0)

        mp_group = None

        inference_config = DeepSpeedInferenceConfig(
            set_empty_params=True,
            dtype=torch.bfloat16,
            max_out_tokens=40,
            min_out_tokens=60,
            transposed_mode=True,
        )

        mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group, mp_size=1, out_dim=0, in_dim=1)

        #
        layer_id = 0

        policy_cls = LLAMALayerPolicy

        policy = policy_cls(module, inference=True)

        _container = policy_to_ds_container(
            policy=policy, config=inference_config, model_config=model_config, layer_id=layer_id, child=module
        )

        # mocker create inference module in hybrid engine
        _container.set_tensor_parallel_config(1, None)
        _container.initialize_tensors(enable_training=True)
        _container.create_ds_model_config()
        _container.create_module()
        _container.set_params_wo_copy(Z3_enabled=True)
        _container.apply_tensor_parallelism(mp_replace, reversed_dim=True)
        _container.module.dtype = torch.bfloat16

        assert torch.allclose(
            torch.cat([policy.client_module.mlp.up_proj.weight, policy.client_module.mlp.gate_proj.weight]),
            _container._h4h_w,
        )

        assert torch.allclose(_container._4hh_w, policy.client_module.mlp.down_proj.weight, rtol=1e-05, atol=1e-08)

        assert torch.allclose(
            _container.module.attention.attn_qw, policy.client_module.self_attn.q_proj.weight, rtol=1e-05, atol=1e-08
        )
        assert torch.allclose(
            _container.module.attention.attn_kw, policy.client_module.self_attn.k_proj.weight, rtol=1e-05, atol=1e-08
        )

        assert torch.allclose(
            _container.module.attention.attn_vw, policy.client_module.self_attn.v_proj.weight, rtol=1e-05, atol=1e-08
        )

        assert torch.allclose(
            _container.module.attention.attn_ow, policy.client_module.self_attn.o_proj.weight, rtol=1e-05, atol=1e-08
        )

        assert torch.allclose(
            _container.module.mlp.inter_up_w, policy.client_module.mlp.up_proj.weight, rtol=1e-05, atol=1e-08
        )
        assert torch.allclose(
            _container.module.mlp.inter_gate_w, policy.client_module.mlp.gate_proj.weight, rtol=1e-05, atol=1e-08
        )
        assert torch.allclose(
            _container.module.mlp.output_w, policy.client_module.mlp.down_proj.weight, rtol=1e-05, atol=1e-08
        )

        assert torch.allclose(
            _container.module.mlp.attn_nw, policy.client_module.input_layernorm.weight, rtol=1e-05, atol=1e-08
        )
        assert torch.allclose(
            _container.module.norm_w, policy.client_module.post_attention_layernorm.weight, rtol=1e-05, atol=1e-08
        )

        rank = 0
        batch_size = 4
        seq_len = 16
        hidden_states = 4096
        rank = 0
        inputs = torch.ones((batch_size, seq_len, hidden_states)).cuda().bfloat16()
        attention_mask = torch.ones((4, 1, 16, 16)).to(rank)

        res1 = policy.client_module.mlp(inputs.bfloat16().to(rank))

        _container.module.attention(inputs.bfloat16().to(rank), attention_mask)
        res2 = _container.module.mlp(inputs.bfloat16().to(rank), None, None, None)

        assert torch.allclose(res1, res2, rtol=1e-05, atol=1e-08)

        res3 = policy.client_module.self_attn(
            inputs.bfloat16().to(rank),
            attention_mask=attention_mask.to(rank),
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        attention_output_container_module, _, _, _, _ = _container.module.attention(
            inputs.bfloat16().to(rank), attention_mask
        )
        assert torch.allclose(res3[0], attention_output_container_module, rtol=1e-05, atol=1e-08)

        res4 = policy.client_module(
            inputs.bfloat16().to(rank),
            attention_mask=attention_mask.to(rank),
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states, past_key_value, _ = _container.module(
            inputs.bfloat16().to(rank),
            attention_mask=attention_mask.to(rank),
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        assert torch.allclose(res4[0], hidden_states, rtol=1e-05, atol=1e-08)

        model = LlamaForCausalLM(model_config)
        model.bfloat16().to(0)

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
            _container.set_tensor_parallel_config(1, None)
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
        tokenizer = AutoTokenizer.from_pretrained("/mnt1/xuantai.hxd/test/Llama-2-7b-hf")
        inputs = tokenizer("凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。", return_tensors="pt")
        rank = 0
        o = model.generate(inputs.input_ids.to(0), use_cache=True, max_new_tokens=50)
        outputs = model.generate(inputs.input_ids.to(0), use_cache=True, max_new_tokens=50)
        assert torch.allclose(outputs, o, rtol=1e-05, atol=1e-08)
        """


if __name__ == "__main__":
    unittest.main()
