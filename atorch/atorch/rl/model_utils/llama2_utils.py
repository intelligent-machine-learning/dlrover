import re

import torch


def get_llama2_params_offsets(config=None, tp_size=1):

    head_coef = config.num_attention_heads // config.num_key_value_heads
    q_proj_start = 0
    q_proj_end = q_proj_start + config.hidden_size * config.hidden_size // tp_size
    k_proj_start = q_proj_end
    k_proj_end = k_proj_start + config.hidden_size * config.hidden_size // tp_size // head_coef
    v_proj_start = k_proj_end

    v_proj_end = v_proj_start + config.hidden_size * config.hidden_size // tp_size // head_coef
    o_proj_start = v_proj_end
    o_proj_end = o_proj_start + config.hidden_size * config.hidden_size // tp_size

    gate_proj_start = o_proj_end
    gate_proj_end = gate_proj_start + config.hidden_size * config.intermediate_size // tp_size

    up_proj_start = gate_proj_end
    up_proj_end = up_proj_start + config.hidden_size * config.intermediate_size // tp_size

    down_proj_start = up_proj_end
    down_proj_end = down_proj_start + config.hidden_size * config.intermediate_size // tp_size

    down_proj_start = up_proj_end
    down_proj_end = down_proj_start + config.hidden_size * config.intermediate_size // tp_size

    input_layernorm_start = down_proj_end
    input_layernorm_end = input_layernorm_start + config.hidden_size

    post_attention_layernorm_start = input_layernorm_end
    post_attention_layernorm_end = post_attention_layernorm_start + config.hidden_size

    offsets_info = {
        "q_proj": [q_proj_start, q_proj_end],
        "k_proj": [k_proj_start, k_proj_end],
        "v_proj": [v_proj_start, v_proj_end],
        "o_proj": [o_proj_start, o_proj_end],
        "gate_proj": [gate_proj_start, gate_proj_end],
        "up_proj": [up_proj_start, up_proj_end],
        "down_proj": [down_proj_start, down_proj_end],
        "input_layernorm": [input_layernorm_start, input_layernorm_end],
        "post_attention_layernorm": [post_attention_layernorm_start, post_attention_layernorm_end],
    }

    return offsets_info


def move_weight_to_continuous_buffer(state_dict, config, offsets_info, tp_size=1, tp_rank=0):
    q_proj_shard_size = config.hidden_size // tp_size
    num_kv_heads_per_gpu = max(1, config.num_key_value_heads // tp_size)
    kv_proj_shard_size = config.hidden_size // config.num_attention_heads * num_kv_heads_per_gpu

    attention_weight_specs = [
        # (weight_name, shard_size, offset)
        ("q_proj", q_proj_shard_size, 0),
        ("k_proj", kv_proj_shard_size, q_proj_shard_size),
        ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
    ]
    param_layer_dict = {}
    param_name_layer_dict = {}
    for name, loaded_weight in state_dict.items():
        layer_id = re.findall(r"\d+", name)
        if len(layer_id) == 0:
            # non decoder layer parameter
            continue
        else:
            layer_id = int(layer_id[0])
            if layer_id not in param_layer_dict.keys():
                param_layer_dict[layer_id] = [loaded_weight]
                param_name_layer_dict[layer_id] = [name]
            else:
                param_layer_dict[layer_id].append(loaded_weight)
                param_name_layer_dict[layer_id].append(name)
    fisrt_layer = sorted([layer_id for layer_id in param_layer_dict.keys()])[0]

    offsets_parameters_num = [p[1] - p[0] for p in offsets_info.values()]
    num_param_per_block = sum(offsets_parameters_num)
    dtype = [p.dtype for p in param_layer_dict[fisrt_layer]][0]
    device = [p.device for p in param_layer_dict[fisrt_layer]][0]
    num_param_decoder_layers = num_param_per_block * len(param_layer_dict.keys())
    param_all_block_decoder_layers = torch.zeros(num_param_decoder_layers, dtype=dtype, device=device)

    for layer_id in param_layer_dict.keys():

        param_start = (layer_id - fisrt_layer) * num_param_per_block
        param_end = (layer_id + 1 - fisrt_layer) * num_param_per_block
        param = param_all_block_decoder_layers.data[param_start:param_end]
        for loaded_weight, name in zip(param_layer_dict[layer_id], param_name_layer_dict[layer_id]):

            if "rotary_emb.inv_freq" in name:
                continue

            is_attention_weight = False

            for weight_name, shard_size, _ in attention_weight_specs:

                if weight_name not in name:
                    continue

                shard_size = loaded_weight.shape[0] // tp_size
                shard_id = tp_rank
                loaded_weight = loaded_weight[shard_size * shard_id : shard_size * (shard_id + 1)]
                param_slice = param.data[offsets_info[weight_name][0] : offsets_info[weight_name][1]]
                param_slice.copy_(loaded_weight.view_as(param_slice))

                is_attention_weight = True
                break

            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                shard_size = loaded_weight.shape[0] // tp_size
                loaded_weight = loaded_weight[shard_size * tp_rank : shard_size * (tp_rank + 1)]
                param_slice = param.data[offsets_info[weight_name][0] : offsets_info[weight_name][1]]
                param_slice.copy_(loaded_weight.view_as(param_slice))
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            for weight_name in ["down_proj", "o_proj"]:
                if weight_name not in name:
                    continue
                shard_size = loaded_weight.shape[1] // tp_size
                start_idx = tp_rank * shard_size
                end_idx = (tp_rank + 1) * shard_size

                loaded_weight = loaded_weight[:, start_idx:end_idx]
                param_slice = param.data[offsets_info[weight_name][0] : offsets_info[weight_name][1]]
                param_slice.copy_(torch.clone(loaded_weight).view_as(param_slice))

            for weight_name in ["input_layernorm", "post_attention_layernorm"]:
                if weight_name not in name:
                    continue
                param_slice = param.data[offsets_info[weight_name][0] : offsets_info[weight_name][1]]
                param_slice.copy_(loaded_weight.view_as(param_slice))
    return param_all_block_decoder_layers
