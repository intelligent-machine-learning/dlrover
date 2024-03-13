import functools
import os
import random
from typing import MutableMapping, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers


def set_seed(seed: int):
    """
    Sets seeds across package dependencies for reproducibility.
    """
    seed += int(os.environ.get("RANK", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []

    glm_embeddings_to_freeze = hf_get_glm_embeddings(model)
    for layer in hidden_layers_to_freeze + glm_embeddings_to_freeze:
        layer.requires_grad_(False)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


# copy from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def rgetattr(obj, attr, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def hf_get_decoder(model: nn.Module) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig, GLMModel)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
        - glm.transformer: (GLMForConditionalGeneration)
    """
    decoder_attrs = (
        "transformer",
        "model.decoder",
        "gpt_neox",
        "decoder",
        "glm.transformer",
    )
    return findattr(model, decoder_attrs)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
        - glm.transformer.final_layernorm: (GLMForConditionalGeneration)
        - transformer.final_layernorm: (GLMModel)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
        "glm.transformer.final_layernorm",
        "transformer.final_layernorm",
    )
    return findattr(model, norm_attrs)


# copy from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def hf_get_decoder_blocks(model):
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
        - glm.transformer.layers: (GLMForConditionalGeneration)
        - transformer.layers: (GLMModel)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
        "glm.transformer.layers",
        "glm.word_embeddings",
        "glm.transformer.position_embeddings",
        "glm.transformer.block_position_embeddings",
        "transformer.layers",
        "word_embeddings",
        "transformer.position_embeddings",
        "transformer.block_position_embeddings",
    )
    return findattr(model, hidden_layers_attrs)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def hf_get_glm_embeddings(model):
    glm_embeddings_attrs = (
        "glm.word_embeddings",
        "glm.transformer.position_embeddings",
        "glm.transformer.block_position_embeddings",
        "word_embeddings",
        "transformer.position_embeddings",
        "transformer.block_position_embeddings",
    )
    all_embeddings = []
    for attr in glm_embeddings_attrs:
        if rhasattr(model, attr):
            all_embeddings.append(rgetattr(model, attr))
    return all_embeddings


# copy from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def hf_get_lm_head(model: nn.Module) -> nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def hf_get_hidden_size(config):
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig, GLMConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def hf_get_num_hidden_layers(config: transformers.PretrainedConfig) -> int:
    """Returns the number of hidden layers in the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different number-of-layers attribute
    names.
        - num_hidden_layers: (GPTNeoXConfig, OPTConfig)
        - n_layer: (GPT2Config, GPTJConfig, BloomConfig)
        - num_layers: (GLMConfig)
    """
    num_hidden_layers_attrs = ("num_hidden_layers", "n_layer", "num_layers")
    return findattr(config, num_hidden_layers_attrs)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
):
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    mean = (xs * mask).sum() / n
    torch_inf = torch.tensor(np.inf, dtype=xs.dtype).to(xs.device)
    return dict(
        mean=mean.item(),
        min=torch.where(mask.bool(), xs, torch_inf).min().item(),
        max=torch.where(mask.bool(), xs, -torch_inf).max().item(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n).item(),
    )


# based on https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs):
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


def is_trainable_model(model_name):
    return "actor" in model_name or "critic" in model_name
