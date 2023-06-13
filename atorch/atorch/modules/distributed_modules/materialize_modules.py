from atorch.utils.meta_model_utils import is_meta, reload_meta_module

from .layers import ATorchTPLayer


def materialize_modules_to_device(model, device):
    # Base case: if the model is an instance of ATorchTPLayer
    if not is_meta(model):
        model.to(device)
    else:
        if isinstance(model, ATorchTPLayer):
            model.reset_parameters()
            model.to(device)
        else:
            # If the model is not an instance of ATorchTPLayer
            # we have to check its submodules
            # and see if any of them are instances of ATorchTPLayer
            has_ATorchTPLayer = any(isinstance(module, ATorchTPLayer) for _, module in model.named_modules())

            # If the model doesn't contain any ATorchTPLayer
            if not has_ATorchTPLayer:
                # We can safely reload the meta modules and move the whole model to the device
                reload_meta_module(model, device)
            else:
                # Otherwise we need to process each child separately
                for name, child in model.named_children():
                    materialize_modules_to_device(child, device)

                if is_meta(model):
                    reload_meta_module(model, device, delete_ckpt_name=False)
