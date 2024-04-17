import functools
import unittest
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import atorch
from atorch.utils.ds_pipe_utils import PipeModuleFromRecordedMeta
from atorch.utils.meta_model_utils import record_module_init
from atorch.utils.version import torch_version


def gpt2_custom_patcher(cfg):
    def wpe_patcher(fw, self):
        @functools.wraps(fw)
        def fw_wrapper(input):
            assert (
                isinstance(input, tuple) and len(input) == 3
            ), "input should be (hidden_states, position_ids, attention_mask)"
            hidden_states, position_ids, attention_mask = input
            position_embeddings = fw(position_ids)
            hidden_states = hidden_states + position_embeddings
            return hidden_states, attention_mask

        return fw_wrapper

    def h_patcher(fw, self):
        @functools.wraps(fw)
        def fw_wrapper(input):
            assert isinstance(input, tuple) and len(input) == 2, "input should be (hidden_states, attention_mask)"
            hidden_states, attention_mask = input
            ori_attn_mask = attention_mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            outputs = fw(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
            return hidden_states, ori_attn_mask

        return fw_wrapper

    gpt2_custom_forward_patchers = {"wpe": wpe_patcher}
    gpt2_custom_forward_patchers.update({f"h.{i}": h_patcher for i in range(cfg.n_layer)})
    return gpt2_custom_forward_patchers


def _weight_align(pipe_model, ref_model):
    sd = ref_model.state_dict()
    prefix_odict = OrderedDict()
    prefix_odict[pipe_model._ori_module_name[0]] = "tied_modules." + pipe_model._ori_module_name[0]
    for i, n in enumerate(pipe_model._ori_module_name[1:]):
        prefix_odict[n + "."] = str(i + 1) + "."
    new_sd = type(sd)()
    for k in sd:
        for prefix in prefix_odict:
            if k.startswith(prefix):
                new_k = k.replace(prefix, prefix_odict[prefix])
                break
        new_sd[new_k] = sd[k]
    pipe_model.load_state_dict(new_sd, strict=False)


class TestDeepspeedPipelne(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch_version() < (2, 0, 0),  # type: ignore
        "skip if no gpu, torch 2.0 needed for torch.device context manager.",
    )
    def test_meta_to_pipelinemodule(self):
        atorch.init_distributed(set_cuda_device_using_local_rank=True)
        gpt2_config = GPT2Config(
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, n_embd=256, n_head=4, n_layer=3, n_positions=512
        )
        with record_module_init():
            meta_model = GPT2Model(gpt2_config)
        device = torch.cuda.current_device()
        pipe_model = PipeModuleFromRecordedMeta(meta_model, gpt2_custom_patcher(gpt2_config), num_stages=1)
        ref_model = GPT2Model(gpt2_config).to(device)
        _weight_align(pipe_model, ref_model)
        input_ids = torch.randint(0, 10000, (2, 512), device=device)
        position_ids = torch.arange(0, 512, dtype=torch.long, device=device)
        attention_mask = torch.randint(0, 2, (2, 512), device=device)
        pipe_out = pipe_model(
            (
                input_ids,
                position_ids,
                attention_mask,
            )
        )
        ref_out = ref_model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        ref_out = F.linear(ref_out.last_hidden_state, ref_model.wte.weight)
        assert torch.all(torch.isclose(pipe_out, ref_out))
        atorch.reset_distributed()


if __name__ == "__main__":
    unittest.main()
