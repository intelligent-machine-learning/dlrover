import functools
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.optim as optim
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from atorch.common.util_func import data_to_device


class ModelType(Enum):
    TOY = auto()
    GPT2 = auto()
    LLAMA = auto()


def get_model_type(name):
    return getattr(ModelType, name.upper(), None)


class ToyModel(nn.Module):
    def __init__(self, in_features=16, out_features=4, num_linears=8):
        super().__init__()
        self.first_linear = nn.Linear(in_features, out_features)
        self.linears = torch.nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_linears - 1)])

    def forward(self, inputs):
        res = self.first_linear(inputs["input"])
        for op in self.linears:
            res = op(res)
        return res


class MyGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        transformer_outputs = super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def vocab_size(self):
        return self.config.vocab_size


def optim_func(model_parameters, **kwargs):
    return optim.Adam(model_parameters, **kwargs)


def toy_loss_func(inputs, output):
    loss = nn.MSELoss()
    return loss(inputs["label"], output)


def gpt2_loss_func(inputs, output, vocab_size):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(output.view(-1, vocab_size), inputs["labels"].view(-1))


def llama_loss_func(inputs, output):
    return output.loss


def prepare_input(data, device):
    return data_to_device(data, device)


def get_model(model_type, config):
    # config: dict with hidden_size, head_num, layer_num, seq_length for llms

    if model_type == ModelType.TOY:
        model = ToyModel(
            in_features=config["in_features"], out_features=config["out_features"], num_linears=config["num_linears"]
        )
        return model

    # llms
    hidden_size = config["hidden_size"]
    head_num = config["head_num"]
    layer_num = config["layer_num"]
    seq_length = config["seq_length"]
    if model_type == ModelType.GPT2:
        model_config = GPT2Config()
        c_s = f"n_embd={hidden_size},n_head={head_num},n_layer={layer_num},n_positions={seq_length}"
        model_config.update_from_string(c_s)
        model = MyGPT2Model(model_config)
    elif model_type == ModelType.LLAMA:
        model_config = LlamaConfig()
        c_s = f"hidden_size={hidden_size},num_attention_heads={head_num},num_hidden_layers={layer_num},"
        c_s += f"num_key_value_heads={head_num},max_position_embeddings={seq_length}"
        model_config.update_from_string(c_s)
        model = LlamaForCausalLM(model_config)
    return model


def get_module_type(model_type):
    if model_type == ModelType.TOY:
        return nn.Linear
    if model_type == ModelType.GPT2:
        return GPT2Block
    if model_type == ModelType.LLAMA:
        return LlamaDecoderLayer
    return None


def get_model_input_format(model_type):
    # get model input format: "unpack_sequence", "unpack_dict", or None.
    if model_type == ModelType.TOY:
        return None
    if model_type == ModelType.GPT2:
        return "unpack_dict"
    if model_type == ModelType.LLAMA:
        return "unpack_dict"
    return None


def get_vocab_size(model_type):
    size = 0
    if model_type == ModelType.GPT2:
        config = GPT2Config()
        size = config.vocab_size
    if model_type == ModelType.LLAMA:
        config = LlamaConfig()
        size = config.vocab_size
    return size


def get_loss_func(model_type):
    if model_type == ModelType.TOY:
        return toy_loss_func
    if model_type == ModelType.GPT2:
        vocab_size = get_vocab_size(model_type)
        return functools.partial(gpt2_loss_func, vocab_size=vocab_size)
    if model_type == ModelType.LLAMA:
        return llama_loss_func
