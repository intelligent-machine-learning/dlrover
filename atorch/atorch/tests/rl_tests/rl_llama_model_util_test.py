import unittest

import torch

# test case 70B
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from atorch.rl.model_utils.llama2_utils import get_llama2_params_offsets, move_weight_to_continuous_buffer


@unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
class TestLLama2Util(unittest.TestCase):
    def test_get_param_offset(self):
        config = LlamaConfig()
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        llama_decoder_layer = LlamaDecoderLayer(config)
        layer_parameters_num = [p.numel() for p in llama_decoder_layer.parameters()]
        offsets_info = get_llama2_params_offsets(config)
        offsets_parameters_num = [p[1] - p[0] for p in offsets_info.values()]
        self.assertListEqual(layer_parameters_num, offsets_parameters_num)
        config.num_hidden_layers = 1
        llama2_casual_lm = LlamaForCausalLM(config)
        state_dict = llama2_casual_lm.state_dict()
        config = llama2_casual_lm.config
        param_tensor = move_weight_to_continuous_buffer(state_dict, config, offsets_info)
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].self_attn.q_proj.weight.flatten(),
            param_tensor[offsets_info["q_proj"][0] : offsets_info["q_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].self_attn.k_proj.weight.flatten(),
            param_tensor[offsets_info["k_proj"][0] : offsets_info["k_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].self_attn.v_proj.weight.flatten(),
            param_tensor[offsets_info["v_proj"][0] : offsets_info["v_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].self_attn.o_proj.weight.flatten(),
            param_tensor[offsets_info["o_proj"][0] : offsets_info["o_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].mlp.gate_proj.weight.flatten(),
            param_tensor[offsets_info["gate_proj"][0] : offsets_info["gate_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].mlp.up_proj.weight.flatten(),
            param_tensor[offsets_info["up_proj"][0] : offsets_info["up_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].mlp.down_proj.weight.flatten(),
            param_tensor[offsets_info["down_proj"][0] : offsets_info["down_proj"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].input_layernorm.weight.flatten(),
            param_tensor[offsets_info["input_layernorm"][0] : offsets_info["input_layernorm"][1]],
        )
        assert torch.allclose(
            llama2_casual_lm.model.layers[0].input_layernorm.weight.flatten(),
            param_tensor[offsets_info["post_attention_layernorm"][0] : offsets_info["post_attention_layernorm"][1]],
        )
        assert param_tensor.numel() == sum(offsets_parameters_num)

    def test_llama_tp(self):
        tp_sizes = [2, 2]
        tp_ranks = [0, 1]
        config = LlamaConfig()
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.num_hidden_layers = 4
        llama2_casual_lm = LlamaForCausalLM(config)
        for p, q in llama2_casual_lm.named_parameters():
            q.requires_grad = False

        trainable_module = llama2_casual_lm.model.layers[-2:]
        for p in trainable_module.parameters():
            p.requires_grad = True
        state_dict = {}
        for p, q in llama2_casual_lm.named_parameters():
            if q.requires_grad is True:
                state_dict[p] = q

        # state_dict = llama2_casual_lm.state_dict()
        offsets_info = get_llama2_params_offsets(config)

        # for rank-0 and rank-1
        for tp_size, tp_rank in zip(tp_sizes, tp_ranks):
            offsets_info = get_llama2_params_offsets(config=config, tp_size=tp_size)
            offsets_parameters_num = [p[1] - p[0] for p in offsets_info.values()]

            trainable_params = move_weight_to_continuous_buffer(
                state_dict, config, offsets_info, tp_size=tp_size, tp_rank=tp_rank
            )

            # for layer -2 and layer -1
            for layer_id in [2, 3]:
                num_param_per_block = sum(offsets_parameters_num)
                start_pos = (layer_id - 2) * num_param_per_block
                param_tensor = trainable_params[start_pos : start_pos + num_param_per_block]
                start = llama2_casual_lm.model.layers[layer_id].self_attn.q_proj.weight.shape[0] // tp_size * tp_rank
                end = (
                    llama2_casual_lm.model.layers[layer_id].self_attn.q_proj.weight.shape[0] // tp_size * (tp_rank + 1)
                )

                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].self_attn.q_proj.weight[start:end].flatten(),
                    param_tensor[offsets_info["q_proj"][0] : offsets_info["q_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )

                start = llama2_casual_lm.model.layers[layer_id].self_attn.k_proj.weight.shape[0] // tp_size * tp_rank
                end = (
                    llama2_casual_lm.model.layers[layer_id].self_attn.k_proj.weight.shape[0] // tp_size * (tp_rank + 1)
                )
                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].self_attn.k_proj.weight[start:end].flatten(),
                    param_tensor[offsets_info["k_proj"][0] : offsets_info["k_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )
                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].self_attn.v_proj.weight[start:end].flatten(),
                    param_tensor[offsets_info["v_proj"][0] : offsets_info["v_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )

                start = llama2_casual_lm.model.layers[layer_id].self_attn.o_proj.weight.shape[1] // tp_size * tp_rank
                end = (
                    llama2_casual_lm.model.layers[layer_id].self_attn.o_proj.weight.shape[1] // tp_size * (tp_rank + 1)
                )

                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].self_attn.o_proj.weight[:, start:end].flatten(),
                    param_tensor[offsets_info["o_proj"][0] : offsets_info["o_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )
                start = llama2_casual_lm.model.layers[layer_id].mlp.gate_proj.weight.shape[0] // tp_size * tp_rank
                end = llama2_casual_lm.model.layers[layer_id].mlp.gate_proj.weight.shape[0] // tp_size * (tp_rank + 1)

                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].mlp.gate_proj.weight[start:end, :].flatten(),
                    param_tensor[offsets_info["gate_proj"][0] : offsets_info["gate_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )
                start = llama2_casual_lm.model.layers[layer_id].mlp.up_proj.weight.shape[0] // tp_size * tp_rank
                end = llama2_casual_lm.model.layers[layer_id].mlp.up_proj.weight.shape[0] // tp_size * (tp_rank + 1)

                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].mlp.up_proj.weight[start:end, :].flatten(),
                    param_tensor[offsets_info["up_proj"][0] : offsets_info["up_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )
                start = llama2_casual_lm.model.layers[layer_id].mlp.down_proj.weight.shape[1] // tp_size * tp_rank
                end = llama2_casual_lm.model.layers[layer_id].mlp.down_proj.weight.shape[1] // tp_size * (tp_rank + 1)

                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].mlp.down_proj.weight[:, start:end].flatten(),
                    param_tensor[offsets_info["down_proj"][0] : offsets_info["down_proj"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )
                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].input_layernorm.weight.flatten(),
                    param_tensor[offsets_info["input_layernorm"][0] : offsets_info["input_layernorm"][1]],
                    rtol=1e-05,
                    atol=1e-08,
                )
                torch.allclose(
                    llama2_casual_lm.model.layers[layer_id].input_layernorm.weight.flatten(),
                    param_tensor[
                        offsets_info["post_attention_layernorm"][0] : offsets_info["post_attention_layernorm"][1]
                    ],
                    rtol=1e-05,
                    atol=1e-08,
                )


if __name__ == "__main__":
    unittest.main()
