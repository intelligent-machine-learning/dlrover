""" Patch Fairscale OSS
OSS optimizer would put trainable params in one bucket
without considering param size. This may result in bad results for model.generate if
some params' sizes (such as param.numel == 1) are not 16-byte aligned.
Patch _setup_flat_buffers in OSS and _add_param_as_view in ParamBucket for alignment.
"""

import torch
from fairscale.nn.misc import ParamBucket
from fairscale.optim.oss import OSS


# Align param address to 16-byte
def _alignment_size(size, element_size):
    alignment = 16
    align_size = alignment // element_size
    return size if size % align_size == 0 else (size // align_size + 1) * align_size


def patch_setup_flat_buffers(self) -> None:
    """Make all params which are on the same device and tied to the same rank views of a single buffer.
    This is used at construction time, and anytime parameter trainability is changed (frozen or unfrozen) and
    `refresh_trainability` is called.
    """

    for device, per_rank_params in self._per_device_params.items():
        # Only wipe the existing buckets if there are none
        # (could be that this is called twice, when trainability changes)
        if device not in self.buckets.keys():
            self.buckets[device] = {}

        # Make parameters a view of the bucket
        for dst_rank, params in enumerate(per_rank_params):
            if len(params) > 0:

                # Clone the non-trainable params, if in a bucket it will get destroyed
                for param in filter(lambda x: not x.requires_grad, params):
                    param.data = param.data.detach().clone()

                # Merge all the trainable params in a single bucket
                trainable_params = list(filter(lambda x: x.requires_grad, params))
                if trainable_params:
                    # align param size
                    buffer_size = sum(map(lambda x: _alignment_size(x.numel(), x.element_size()), trainable_params))
                    bucket = ParamBucket(size=buffer_size, dtype=trainable_params[0].dtype, device=device)

                    for param in trainable_params:
                        bucket.add_param(param)

                    self.buckets[device][dst_rank] = bucket

    # Clear the buffer keys which are not in use anymore (could be that the devices changed)
    devices_in_use = list(self._per_device_params.keys())
    devices_to_pop = list(filter(lambda x: x not in devices_in_use, self.buckets.keys()))
    for d in devices_to_pop:
        self.buckets.pop(d)


@torch.no_grad()
def patch_add_param_as_view(self, param: torch.Tensor, keep_existing_value: bool = True) -> None:
    assert self.buffer is not None
    assert (
        param.dtype == self.buffer.dtype
    ), f"Different types for the bucket and the param, cannot proceed: {param.dtype} - {self.buffer.dtype}"
    assert (
        param.device == self.buffer.device
    ), f"Different devices for the bucket and the param, cannot proceed: {param.device} - {self.buffer.device}"

    fill_next = self._fill + param.numel()
    assert fill_next <= self.buffer.numel()

    # Copy the current param value
    if keep_existing_value:
        self.buffer[self._fill : fill_next].copy_(param.data.flatten())
    param.data = self.buffer[self._fill : fill_next].view_as(param.data)
    # align param size
    self._fill = _alignment_size(fill_next, param.element_size())


OSS._setup_flat_buffers = patch_setup_flat_buffers
ParamBucket._add_param_as_view = patch_add_param_as_view
