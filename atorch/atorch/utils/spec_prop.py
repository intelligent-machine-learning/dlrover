import gc
import traceback

import torch
from torch.fx.node import Node, map_aggregate

from atorch.utils.meta_model_utils import move_args_kwargs_to_device, reload_meta_module
from atorch.utils.sharding_spec import _extract_sharding_spec
from atorch.utils.version import torch_version


class SpecProp(torch.fx.Interpreter):
    def __init__(self, gm, fake_mode=None):
        super().__init__(gm)
        if fake_mode is None and torch_version() > (2, 0, 0):
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor

            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module

    def run_node(self, n):
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode:
                        result = super().run_node(n)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module

            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            inputs = list(args) + list(kwargs.values())
            output_sharding_spec = map_aggregate(result, _extract_sharding_spec)
            n.meta["output_sharding_spec"] = output_sharding_spec
            input_sharding_spec = map_aggregate(inputs, _extract_sharding_spec)
            keys = list(n.args) + list(n.kwargs.values())
            # example:
            # code: (self.layer(inp), ) + (1, )
            # graph: add, args: ((layer, ), (1, ))
            # We need to unwrap the sharding spec here
            input_specs = dict()

            def _unwrap_sharding_spec(node_arg, sharding_spec):
                if isinstance(node_arg, Node):
                    input_specs[node_arg.name] = sharding_spec
                elif isinstance(node_arg, (list, tuple)):
                    for n_arg, s_spec in zip(node_arg, sharding_spec):
                        _unwrap_sharding_spec(n_arg, s_spec)
                elif isinstance(node_arg, dict):
                    for n_arg, s_spec in zip(node_arg.values(), sharding_spec.values()):
                        _unwrap_sharding_spec(n_arg, s_spec)

            for i, key in enumerate(keys):
                _unwrap_sharding_spec(key, input_sharding_spec[i])

            n.meta["input_sharding_spec"] = input_specs

            return result
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"SpecProp error for: node={n.format_node()} with " f"meta={n.meta}") from e

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
            fake_args = [self.fake_mode.from_tensor(t) for t in args]
        else:
            fake_args = args
        return super().run(*fake_args)


class MetaSpecProp(SpecProp):
    """Propagate the sharding spec through the Graph

    Args:
        *args: input to the GraphModule

    Return:
        None. Each node in the GraphModule's graph will have a new attribute meta
            meta['output_sharding_spec'] is of the same type as the output of the node,
            meta['input_sharding_spec'] is a dict, keys being the input node's name
            values being the corresponding sharding spec
    """

    def __init__(self, module, device="cpu"):
        self._device = device
        super().__init__(module)
        self.fake_module = None
        self.fake_mode = None

    def call_module(self, target, args, kwargs):
        # test if is call_module
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        reload_meta_module(submod, device=self._device, delete_ckpt_name=False)
        args, kwargs = move_args_kwargs_to_device(args, kwargs, self._device)
        with torch.no_grad():
            res = submod(*args, **kwargs)

        orig_module = submod
        orig_res = res
        submod = submod.to("cpu")
        args, kwargs = move_args_kwargs_to_device(args, kwargs, "cpu")
        res, _ = move_args_kwargs_to_device(res, {}, "cpu")
        del orig_res
        del orig_module
        del args
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()
        return res

    def call_function(self, target, args, kwargs):
        # test if is call_module
        assert not isinstance(target, str)
        args, kwargs = move_args_kwargs_to_device(args, kwargs, self._device)
        # Execute the function and return the result
        with torch.no_grad():
            res = target(*args, **kwargs)
        args, kwargs = move_args_kwargs_to_device(args, kwargs, "cpu")
        orig_res = res
        res, _ = move_args_kwargs_to_device(res, {}, "cpu")
        del orig_res
        del args
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()
        return res

    def call_method(self, target, args, kwargs):
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        # test if is call_module
        assert isinstance(target, str)
        args, kwargs = move_args_kwargs_to_device(args, kwargs, self._device)
        with torch.no_grad():
            res = getattr(self_obj, target)(*args_tail, **kwargs)
        args, kwargs = move_args_kwargs_to_device(args, kwargs, "cpu")
        orig_res = res
        res, _ = move_args_kwargs_to_device(res, {}, "cpu")
        del orig_res
        del args
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()
        return res
