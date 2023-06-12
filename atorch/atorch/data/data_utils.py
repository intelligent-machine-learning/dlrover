import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from atorch.common.util_func import recursively_apply


def fast_batch_copy(data):
    def do_copy(data):
        return torch.from_numpy(data.numpy().copy())

    return recursively_apply(do_copy, data, error_on_other_type=True)


def get_sample_batch(dataset, dataloader_args, num=1):
    new_args = {"num_workers": 0}
    for key in dataloader_args:
        if key in ["num_workers", "prefetch_factor"]:
            # multi-process is not needed.
            continue
        if key == "sampler" and type(dataloader_args["sampler"]) == DistributedSampler:
            # no need for sampler if it is default
            continue
        new_args[key] = dataloader_args[key]
    dataloader = DataLoader(dataset, **new_args)
    batches = []
    for idx, data in enumerate(dataloader):
        batches.append(data)
        if idx == num - 1:
            break
    if num == 1:
        return batches[0]
    else:
        return batches


def expand_batch_dim(data, batch_size=1):
    def expand(data, batch_size=1):
        shape_list = list(data.shape)
        shape_list.insert(0, batch_size)
        return data.expand(*shape_list)

    return recursively_apply(expand, data, batch_size=batch_size, error_on_other_type=False)
