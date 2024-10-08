import inspect

import torch
import torch.nn.functional as F

from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import divide, is_wrapped_by_context_manager
from atorch.distributed.distributed import parallel_group, parallel_group_size, rank
from atorch.modules.distributed_modules.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from atorch.modules.distributed_modules.mappings import copy_to_group
from atorch.modules.distributed_modules.randomizer import get_randomizer
from atorch.modules.transformer.layers import dropout_add_layer_norm, flash_attn


def hf_init_weights_custom_fn(meta_model):
    """
    HuggingFace PreTrainedModel applies _init_weights post init.
    We wrap _init_weights operation to custom _build_fn for rebuild from leaf module.

    Warning::
    - If `_init_weights` has checking on parameter with name path or specific module, building leaf module and
      call _init_weights will not be semantically identical with root PretrainedModel applying _init_weights.
    """
    assert hasattr(meta_model, "_init_weights"), "HuggingFace PreTrainedModel should have '_init_weights'."

    def custom_fn_helper(module):
        def _build_fn(*args, **kwargs):
            builded_module = module.__class__(*args, **kwargs)
            builded_module.apply(meta_model._init_weights)
            return builded_module

        # module with param/buffer need _init_weights
        if len(module._parameters) != 0 or len(module._buffers) != 0:
            module._build_fn = _build_fn

    meta_model.apply(custom_fn_helper)
    return meta_model


def tp_shard_helper(meta_module, tp_layer_cls, **tp_kwargs):
    """
    Custom _build_fn to shard tensor parallel Linear/Embedding. The original build fn will be under tensor
    group randomizer to get the same master weight.

    Arguments::
        - meta_module: the meta module requires _build_fn to hook tensor parallel sharding.
        - tp_layer_cls: ATorchTPLayer successors. e.g. ColumnParallelLinear.
        - tp_kwargs: keyword arguments for tp_layer_cls.
    """
    ori_build_fn = meta_module._build_fn if hasattr(meta_module, "_build_fn") else meta_module.__class__

    def _build_fn(*args, **kwargs):
        # embedding size divided by tensor parallel size
        if isinstance(meta_module, torch.nn.Embedding):
            bound_args = inspect.signature(torch.nn.Embedding).bind(*args, **kwargs)
            args, kwargs = bound_args.args, bound_args.kwargs
            num_embeddings, tp_size = args[0], parallel_group_size("tensor")
            padded_num_embeddings = num_embeddings + (-num_embeddings) % tp_size
            args = (
                padded_num_embeddings,
                *args[1:],
            )

        # init master weight in same randomizer
        with get_randomizer("tensor", "data").fork():
            builded_module = ori_build_fn(*args, **kwargs)

        # gpt2 Conv1D compat
        from transformers.pytorch_utils import Conv1D

        if isinstance(builded_module, Conv1D):
            builded_module.in_features, builded_module.out_features = builded_module.weight.shape
            builded_module.weight.data = builded_module.weight.t()

        builded_module = tp_layer_cls(orig_module=builded_module, **tp_kwargs)
        return builded_module

    return _build_fn


def randomizer_helper(*same_groups):
    """
    Custom _post_fn to wrap randomizer contextmanager for builded module's `forward`
    """

    def _post_fn(builded_module):
        builded_module.forward = get_randomizer(*same_groups).fork()(builded_module.forward)
        return builded_module

    return _post_fn


def shrink_attr_helper(attrs):
    """
    Custom _post_fn to shrink the specific attributes for tensor parallel. e.g. num_heads.
    """

    def _post_fn(builded_module):
        for attr in attrs:
            shrinked_value = divide(getattr(builded_module, attr), parallel_group_size("tensor"))
            setattr(builded_module, attr, shrinked_value)
        return builded_module

    return _post_fn


class TPInfo:
    """
    Manual tensor parallel information class.

    Example:
        >>> gpt2_tpinfo = TPInfo()
        >>> gpt2_tpinfo.shard_col({"attn.c_attn": {"stride": 3}}, "mlp.c_fc")
        >>> gpt2_tpinfo.shard_row("attn.c_proj", "mlp.c_proj")
        >>> gpt2_tpinfo.shard_vocab("wte")
        >>> gpt2_tpinfo.replic_drop("resid_dropout", "mlp.dropout", "drop")
        >>> gpt2_tpinfo.parallel_drop("attn_dropout")
        >>> gpt2_tpinfo.shrink({".attn": {"embed_dim", "split_size", "num_heads"}})
        >>> tp_manual_shard_custom_fn(meta_gpt2, gpt2_tpinfo)
    """

    def __init__(self):
        self.Shard = {ColumnParallelLinear: dict(), RowParallelLinear: dict(), VocabParallelEmbedding: dict()}
        self.Drop = {
            ("tensor",): set(),  # replicate dropout
            tuple(): set(),  # parallel dropout
        }
        self.Shrink = dict()

    def parse_args(self, *args):
        ret = dict()
        for arg in args:
            if isinstance(arg, str):
                ret[arg] = dict()
            elif isinstance(arg, dict):
                ret.update(arg)
            else:
                raise ValueError(f"Arg {arg} is not str or dict.")
        return ret

    def shard_col(self, *args):
        self.Shard[ColumnParallelLinear].update(self.parse_args(*args))

    def shard_row(self, *args):
        self.Shard[RowParallelLinear].update(self.parse_args(*args))

    def shard_vocab(self, *args):
        self.Shard[VocabParallelEmbedding].update(self.parse_args(*args))

    def replic_drop(self, *args):
        assert all(isinstance(arg, str) for arg in args), f"Args {args} should all be str."
        self.Drop[("tensor",)].update(args)

    def parallel_drop(self, *args):
        assert all(isinstance(arg, str) for arg in args), f"Args {args} should all be str."
        self.Drop[tuple()].update(args)

    def shrink(self, *args):
        self.Shrink.update(self.parse_args(*args))

    def is_vocab_parallelled(self):
        return len(self.Shard[VocabParallelEmbedding]) > 0


def tp_manual_shard_custom_fn(meta_model, tpinfo):
    # maybe wrap randomizer for flash attn ops; patch _forward fn which called indirectly
    # check if wrapped to avoid repeated wrapping
    if flash_attn is not None:
        for fn_name in dir(flash_attn.flash_attn_interface):
            fn = getattr(flash_attn.flash_attn_interface, fn_name)
            if fn_name.endswith("forward") and not is_wrapped_by_context_manager(fn):
                randomized_fn = get_randomizer().fork()(fn)
                setattr(flash_attn.flash_attn_interface, fn_name, randomized_fn)
    if dropout_add_layer_norm is not None:
        fn_name = "_dropout_add_layer_norm_forward"
        fn = getattr(flash_attn.ops.layer_norm, fn_name)
        if not is_wrapped_by_context_manager(fn):
            randomized_fn = get_randomizer("tensor").fork()(fn)
            setattr(flash_attn.ops.layer_norm, fn_name, randomized_fn)

    # hook _post_fn/_build_fn for tensor parallel
    registry_dict = dict()
    for name, module in meta_model.named_modules():
        # tp shard module
        for tp_layer_cls, shard_suffix in tpinfo.Shard.items():
            for suffix, tp_kwargs in shard_suffix.items():
                if name.endswith(suffix):
                    registry_dict[name] = f"[tp_shard] {tp_layer_cls.__name__}, {tp_kwargs}"
                    module._build_fn = tp_shard_helper(module, tp_layer_cls, **tp_kwargs)

        # dropout randomizer
        for same_groups, drop_suffix in tpinfo.Drop.items():
            if any(name.endswith(suffix) for suffix in drop_suffix):
                registry_dict[name] = f"[randomizer] {same_groups}"
                module._post_fn = randomizer_helper(*same_groups)

        # shrink attribute
        for suffix, attrs in tpinfo.Shrink.items():
            if name.endswith(suffix):
                registry_dict[name] = f"[shrink_attr] {attrs}"
                module._post_fn = shrink_attr_helper(attrs)
    _print_tp_tree(registry_dict)


def _print_tp_tree(registry_dict):
    class node(object):
        def __init__(self, value, children=None):
            self.value = value
            if children is None:
                self.children = dict()

        def __str__(self, level=0):
            ret = "  " * level + repr(self.value) + "\n"
            if len(self.children) > 0 and all(name.isnumeric() for name in self.children):
                child_ret = dict()
                for child in self.children.values():
                    child_str = child.__str__(level + 1)
                    lines = child_str.split("\n")
                    first_line, child_content = lines[0], "\n".join(lines[1:])
                    # coalesced child with same str
                    child_ret[child_content] = child_ret.get(child_content, []) + [first_line.strip()]
                for child_content in child_ret:
                    layer_name_lst = child_ret[child_content]
                    ret += "  " * (level + 1) + f"[{layer_name_lst[0]}...{layer_name_lst[-1]}]\n" + child_content
            else:
                for child in self.children.values():
                    ret += child.__str__(level + 1)
            return ret

    root = node("model")
    for name, content in registry_dict.items():
        path_lst, cur_node = name.split("."), root
        for cur_name in path_lst:
            if cur_name not in cur_node.children:
                cur_node.children[cur_name] = node(cur_name)
            cur_node = cur_node.children[cur_name]
        cur_node.value += " -> " + content
    if rank() == 0:
        logger.info(("Tensor Parallel Sharding Tree:\n" + str(root)).replace('"', "").replace("'", ""))


def vocab_parallel_logit_helper(embed, lm_output):
    """
    Similar to `_default_logit_helper`, but copy lm output to group
    """
    if isinstance(lm_output, torch.Tensor):
        lm_output = lm_output
    elif isinstance(lm_output, tuple):
        lm_output = lm_output[0]
    else:
        raise ValueError(f"Expect lm_output as tensor or tuple but get {type(lm_output)}")

    # VocabParllelEmbedding needs copy lm output to group
    lm_output = copy_to_group(lm_output, group=parallel_group("tensor"))
    return F.linear(lm_output, embed.weight)
