import gc

import torch
from torch.fx.passes.shape_prop import ShapeProp

from atorch.utils.meta_model_utils import move_args_kwargs_to_device, reload_meta_module
from atorch.utils.version import torch_version


class MetaShapeProp(ShapeProp):
    """This is shape propagation method that supports offloading with meta model
    It works exactly as the original ShapeProp class, but a submodule is immediately offloaded
    to disk.
    """

    def __init__(self, module, device="cpu"):
        self._device = device
        if torch_version() < (2, 0, 0):
            super().__init__(module)
        else:
            # Avoid using fake mode until this is fully understood
            super().__init__(module, fake_mode=None)

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
