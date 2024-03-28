import os
from collections import Counter
from contextlib import contextmanager

import torch


def move_to_device(args_or_kwargs, device):
    if isinstance(args_or_kwargs, (list, tuple)):
        return [move_to_device(arg, device) for arg in args_or_kwargs]
    elif isinstance(args_or_kwargs, dict):
        return {k: move_to_device(v, device) for k, v in args_or_kwargs.items()}
    elif isinstance(args_or_kwargs, torch.Tensor):
        return args_or_kwargs.to(device)
    else:
        return args_or_kwargs


def move_to_same_device(args_or_kwargs1, args_or_kwargs2):

    if isinstance(args_or_kwargs1, (list, tuple)):  # they can be list or tuple
        return [move_to_same_device(arg1, arg2) for arg1, arg2 in zip(args_or_kwargs1, args_or_kwargs2)]
    assert type(args_or_kwargs1) is type(args_or_kwargs2), "type not same: %s,%s" % (
        type(args_or_kwargs1),
        type(args_or_kwargs2),
    )
    if isinstance(args_or_kwargs1, dict):
        return {
            k: move_to_same_device(v1, v2) for (k, v1), (_, v2) in zip(args_or_kwargs1.items(), args_or_kwargs2.items())
        }
    elif isinstance(args_or_kwargs1, torch.Tensor):
        return args_or_kwargs1.to(args_or_kwargs2.device)
    else:
        return args_or_kwargs1


no_allclose = {
    # funcname: err_msg
}
type_counter: Counter = Counter()


def check_allclose_and_save(tensor_saved, tensor, name, counter, atol=0, rtol=1e-2):
    saved_output_cpu = move_to_device(tensor_saved, "cpu")
    output_cpu = move_to_device(tensor, "cpu")
    try:
        torch.testing.assert_close(saved_output_cpu, output_cpu)
        return True
    except AssertionError as e:
        no_allclose["%s.%d" % (name, counter)] = e
        return False


def patch_module(old_module, save_dir, name, mode="save", atol=0, rtol=1e-2):
    # save forward and load at next time,check if they are same
    os.makedirs(save_dir, exist_ok=True)
    if mode == "save":
        counter = 0

        def new_forward(*args, **kwargs):
            out = old_forward(*args, **kwargs)
            nonlocal counter
            counter += 1
            filename = os.path.join(save_dir, "%s.check_numberic.idx=%d.pth" % (name, counter))
            to_save = {
                "module": old_module.state_dict(),
                "output": move_to_device(out, "cpu"),
                "args": move_to_device(args, "cpu"),
                "kwargs": move_to_device(kwargs, "cpu"),
            }
            torch.save(to_save, filename)
            return out

        old_forward = old_module.forward
        old_module.forward = new_forward
        old_module.old_forward = old_forward
    else:
        # load
        counter = 0

        def new_forward(*args, **kwargs):
            out = old_forward(*args, **kwargs)
            nonlocal counter
            counter += 1
            filename = os.path.join(save_dir, "%s.check_numberic.idx=%d.pth" % (name, counter))

            if not os.path.exists(filename):
                print("Warning: %s not exists" % filename)
                return out
            saved = torch.load(filename)

            b1 = check_allclose_and_save(saved["output"], out, name, counter, atol, rtol)
            if not b1:
                # move saved args and kwargs,do output check
                device_args = move_to_same_device(saved["args"], args)
                device_kwargs = move_to_same_device(saved["kwargs"], kwargs)
                device_output_with_saved_input = old_forward(*device_args, **device_kwargs)
                check_allclose_and_save(
                    saved["output"], device_output_with_saved_input, name + ".check", counter, atol, rtol
                )

                type_counter[type(old_module)] += 1
            return out

        old_forward = old_module.forward
        old_module.forward = new_forward
        old_module.old_forward = old_forward


@contextmanager
def module_numberic_checker(model, mode="save", dir="./logs/check_numberic"):
    """
    Usage:
    >>> with module_numberic_checker(model, mode="save"):
            model.to("cpu")
            img_feats = model.infer_image(img_data)["cls_vlffn_feats"]
            txt_feats = model.infer_text(txt_data)["cls_vlffn_feats"]
        model.to("npu")

        with module_numberic_checker(model, mode="load") as no_allclose:
            img_feats = model.infer_image(img_data)["cls_vlffn_feats"]
            txt_feats = model.infer_text(txt_data)["cls_vlffn_feats"]
        for func_name, err_msg in no_allclose.items():
            print(func_name, err_msg)
    func_name like: `classifier.1` means the first time `classifier` called,
                    `classifier.2.check` means the second time `classifier` called with saved input
    this two categories of error:
    1. `classifier.1 Tensor-likes are not close!` means the output of `classifier` called
        at first time is not close to saved output with **same model input**
    2. `classifier.2.check Tensor-likes are not close!` : means the output of `classifier` called
        at second time is not close to saved output with **saved input**
        —— show that model parameters are changed or same parameters but different output with
        different device (npu/gpu)


    """

    assert mode in ["save", "load"]
    if mode == "load":
        no_allclose.clear()
    for name, module in model.named_modules():
        if hasattr(module, "forward"):
            patch_module(module, dir, name, mode=mode)
    if mode == "load" and no_allclose:
        for key, error_msg in no_allclose.items():
            print("key not allclose:", key, error_msg)
    yield no_allclose
    for name, module in model.named_modules():
        if hasattr(module, "forward"):
            module.forward = module.old_forward
