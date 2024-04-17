from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import ModelOutput

from atorch.rl.model_utils.model_util import (
    freeze_bottom_causal_layers,
    hf_get_decoder_blocks,
    hf_get_decoder_final_norm,
    hf_get_hidden_size,
    make_head,
)


@dataclass
class ActorCriticRefOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    ref_logits: Optional[torch.FloatTensor] = None
    values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class ActorCriticRef(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.checkpoint_activations = model.config.checkpoint_activations
        self.num_layers = num_layers

        # ref model
        decoder_blocks = hf_get_decoder_blocks(model)
        self.decoder_blocks = deepcopy(nn.ModuleList(list(decoder_blocks)[-num_layers:]))
        self.final_norm = hf_get_decoder_final_norm(model)
        self.word_embeddings_weight = model.glm.word_embeddings.weight

        # critic model
        self.critic_head = make_head(hf_get_hidden_size(model.config), 1)

    def forward(self, *args, inference=False, **kwargs):
        """
        calculating logits, value and ref_logits

        Args:
            generate: bool, when it's True, forward is used for generating response sequenece
            inference: bool, when it's True, forward is used for calculating logits, ref_logits and values
            When generate and inference are both False, forward is used for training and calculating logits and values
        """
        logits = None
        ref_logits = None
        values = None
        outs = self.model.glm(*args, **kwargs)
        logits = outs.logits
        values = self.critic_head(outs.last_hidden_states).squeeze(-1)
        if inference:
            ref_logits, _ = self.get_ref_logits(outs.mems, **inputs)
        return ActorCriticRefOutput(logits=logits, ref_logits=ref_logits, values=values)

    def generate(self, *args, **kwargs):

        """
        feeded with prompts and generating response sequences
        """
        return self.model.generate(*args, **kwargs)

    def get_ref_logits(self, hidden_states, **forward_kwargs):
        input_hidden_state = hidden_states[-(self.num_layers + 1)]
        output_shape = hidden_states[-1].size()
        forward_kwargs.pop("input_ids", None)  # Ignore `input_ids` for branch head
        forward_kwargs.pop("inputs_embeds", None)  # Ignore `inputs_embeds` for branch head
        forward_kwargs.pop("token_type_ids", None)
        input_hidden_state = input_hidden_state.type(dtype=self.word_embeddings_weight.dtype)
        outputs = self.ref_forward(input_hidden_state, output_shape, **forward_kwargs)
        return outputs

    def ref_forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        memory_states=None,
    ):
        def check_detach(_hidden_states):
            return _hidden_states.detach()

        mem_layers = [check_detach(hidden_states)]
        for i, layer in enumerate(self.decoder_blocks):
            args = [hidden_states, attention_mask]

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            mem_i = memory_states[i] if memory_states else None
            if self.checkpoint_activations:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mem=mem_i,
                )
            else:
                hidden_states = layer(*args, mem=mem_i)
            mem_layers.append(check_detach(hidden_states))
        output = self.final_norm(hidden_states)
        mem_layers = self.update_mems(mem_layers, memory_states)

        logits = F.linear(output, self.word_embeddings_weight)
        # to do import CausalLMOutputWithValue
        return logits, mem_layers

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length

        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length :], hiddens[i]), dim=1))
        return new_mems

    @classmethod
    def from_pretrained(cls, path, num_layers=0):
        model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True)
        freeze_bottom_causal_layers(model, num_layers)
        return cls(model, num_layers)


if __name__ == "__main__":
    model_name = "/w/glm-large-chinese"
    actor_ref_critic = ActorCriticRef.from_pretrained(model_name, num_layers=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    str = ["今天天气真好 [MASK]"]
    inputs = tokenizer(str)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v)
    # default inference = False
    outs = actor_ref_critic(**inputs)
    logits = outs.logits
    assert logits.dim() == 3

    # inference = True
    outs = actor_ref_critic(**inputs, inference=True)

    assert (
        torch.norm(outs.ref_logits - outs.logits, 1).detach().numpy() == 0.0
    ), "logits calculated from input_ids and hidden states shoule equal"
    assert outs.values.dim() == 2

    # inference = False
    outs = actor_ref_critic(**inputs, inference=False)
    assert outs.ref_logits is None
