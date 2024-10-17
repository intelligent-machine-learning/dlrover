import functools
import inspect
from collections import Counter

import torch
import torch.nn.functional as F

try:
    import deepspeed
    from deepspeed.pipe import LayerSpec, PipelineModule, TiedLayerSpec
except (ImportError, ModuleNotFoundError):
    deepspeed = None
    LayerSpec = PipelineModule = TiedLayerSpec = object

from atorch.common.log_utils import default_logger as logger
from atorch.distributed import is_distributed, rank
from atorch.distributed.distributed import _DistributedContext as dc
from atorch.utils.meta_model_utils import build_recorded_module


def _default_forward_patcher(forward_fn, self):
    """
    Patch the pipeline layers' ``forward``. Many modules (embed, layernorm, etc.)
    take one or fewer inputs, but we must convey through all the needed tensors to satisfy
    pipeline send/recv mechanism, e.g. attention mask in transformer-like models.
    The default patcher would take the previous tensors to match fn signature, and
    replace them to output the same numbers of tensors.
    One can customize patch_fn in `RecordedMetaLayerSpec` or `RecordedMetaTiedLayerSpec`
    if there is extra compute logic or different input/output format.

    note:
        Deepspeed pipeline engine only supports passing a tensor or a tuple of tensors, while
        those in `float` dtype must requires grad (thus attn mask must in `int`)
    """
    f_sig = inspect.signature(forward_fn)
    input_len = len(f_sig.parameters)

    @functools.wraps(forward_fn)
    def wrapper(inputs, **kwargs):
        assert (
            isinstance(inputs, (torch.Tensor, tuple)) and len(kwargs) == 0
        ), "deepspeed pipeline should only pass tensor or tuple of tensors"
        if isinstance(inputs, torch.Tensor):
            output = forward_fn(inputs)
        else:
            output = forward_fn(*inputs[:input_len])
        if not isinstance(output, tuple):
            output = (output,)
        return *output, *inputs[len(output) :]

    return wrapper


class RecordedMetaLayerSpec(LayerSpec):
    """
    Derived LayerSpec supports building module from the recorded meta module and
    patching the module's ``forward`` with a default patcher or a optional patch_fn
    """

    def __init__(self, meta_module, patch_fn=None):
        self.meta_module = meta_module
        self.patch_fn = patch_fn if patch_fn else _default_forward_patcher
        super().__init__(meta_module.__class__, *meta_module._init_args, **meta_module._init_kwargs)

    def build(self, log=False):
        module = build_recorded_module(self.meta_module)
        module.forward = self.patch_fn(module.forward, module)
        return module


class RecordedMetaTiedLayerSpec(TiedLayerSpec):
    """
    Derived TiedLayerSpec similar to `RecordedMetaLayerSpec`. Note that patch_fn
    will take effect only if forward_fn (usually served as logit_helper) is None
    """

    def __init__(self, key, meta_module, forward_fn=None, patch_fn=None, tied_weight_attr="weight"):
        self.meta_module = meta_module
        self.patch_fn = patch_fn if patch_fn else _default_forward_patcher
        super().__init__(
            key,
            meta_module.__class__,
            *meta_module._init_args,
            forward_fn=forward_fn,
            tied_weight_attr=tied_weight_attr,
            **meta_module._init_kwargs,
        )

    def build(self, log=False):
        module = build_recorded_module(self.meta_module)
        if self.forward_fn is None:
            # without fw_fn, it's the first vocab embedding layer
            module.forward = self.patch_fn(module.forward, module)
        return module


class PipeModuleFromRecordedMeta(PipelineModule):
    """
    Establish PipelineModule from recorded meta model.
    Arguments:
    - meta_model: the recorded meta model constructed under `record_module_init`
    - custom_patcher: Optional dict, the modules that name in keys will assign the value as
                      patch_fn when building LayerSpec
    - tie_first: If True, the first module will be considered as tied embedding layer, and a
                 corresponding logit layer will be appended. Especially for Transformer model
    - logit_helper: The helper function to the tied embedding. If not configured, a default one
                    will be used. Take effect when tie_first is True.
    - kwargs: same as `PipelineModule`
    """

    def __init__(self, meta_model, custom_patcher=None, tie_first=True, logit_helper=None, **kwargs):
        self.tie_first = tie_first
        meta_module_list = self.model_to_module_list(meta_model)
        self._ori_module_name = [n for n, m in meta_module_list]
        # the most common cls name is the Layer block, should configured in partition_method/checkpointable_layers
        self.layerblock_cls_name, self.num_layerblock = Counter(
            [m[1].__class__.__name__ for m in meta_module_list]
        ).most_common()[0]
        if custom_patcher is None:
            if is_distributed() and rank() == 0 or not is_distributed():
                logger.warning(
                    f"No custom_patcher for modules {self._ori_module_name}, "
                    f"make sure _default_forward_patcher satisfy"
                )
        layer_speces = self.convert_to_layer_speces(meta_module_list, custom_patcher, tie_first, logit_helper)

        # ds commu group needs init
        assert dc.INITIALIZED, "Should call atorch.init_distributed first."
        deepspeed.init_distributed()

        kwargs.update({"partition_method": "type:" + self.layerblock_cls_name})
        kwargs.update({"checkpointable_layers": [self.layerblock_cls_name]})
        super(PipeModuleFromRecordedMeta, self).__init__(layers=layer_speces, **kwargs)

    @staticmethod
    def model_to_module_list(model):
        """
        traverse through model modules and convert to sequential module list,
        containing (name, module) of layer modules and leaf notlayer modules.

        Note: Only sequential model is supported, and the definition order of
              modules needs to be the same as the calculation order
        """
        module_list = []

        def _traverse(cur_module, cur_name=""):
            if isinstance(cur_module, (torch.nn.modules.container.ModuleList, torch.nn.modules.container.Sequential)):
                for i, layer in enumerate(cur_module):
                    module_list.append((f"{cur_name}.{i}", layer))
            elif len(cur_module._modules) == 0:
                module_list.append((cur_name, cur_module))
            else:
                for subname, submodule in cur_module._modules.items():
                    if submodule is None:
                        continue
                    full_subname = cur_name + ("." if cur_name else "") + subname
                    _traverse(submodule, full_subname)

        _traverse(model)
        return module_list

    @staticmethod
    def convert_to_layer_speces(module_list, custom_patcher=None, tie_first=True, logit_helper=None):
        """
        Establish layer_spec for pipelinemodule. See `PipeModuleFromRecordedMeta` for args.
        """
        layer_speces = []
        if custom_patcher is None:
            custom_patcher = dict()

        if tie_first:
            # the first module should be tied embedding module, leaf module's tied weight is already 'weight'
            embed_name, embed_module = module_list[0]
            layer_speces.append(
                RecordedMetaTiedLayerSpec(
                    embed_name, embed_module, tied_weight_attr="weight", patch_fn=custom_patcher.get(embed_name, None)
                )
            )

        # the rest modules
        for name, module in module_list[1:] if tie_first else module_list:
            layer_speces.append(RecordedMetaLayerSpec(module, patch_fn=custom_patcher.get(name, None)))

        # tied embed logit_helper
        def _default_logit_helper(embed, lm_output):
            if isinstance(lm_output, torch.Tensor):
                lm_output = lm_output
            elif isinstance(lm_output, tuple):
                lm_output = lm_output[0]
            else:
                raise ValueError(f"Expect lm_output as tensor or tuple but get {type(lm_output)}")
            return F.linear(lm_output, embed.weight)

        if tie_first:
            layer_speces.append(
                RecordedMetaTiedLayerSpec(
                    embed_name,
                    embed_module,
                    tied_weight_attr="weight",
                    forward_fn=logit_helper if logit_helper is not None else _default_logit_helper,
                )
            )

        return layer_speces
