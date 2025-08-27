import torch
import torch.utils
import torch.utils.data
import torch_musa

def patch_after_import_torch():
    # 1. Patch for torch.xxx
    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.current_device = lambda : f'musa:{torch.musa.current_device()}'
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.Event = torch.musa.Event
    torch.cuda.current_stream = torch.musa.current_stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties
    if hasattr(torch.musa, 'is_bf16_supported'):
        torch.cuda.is_bf16_supported = torch.musa.is_bf16_supported
    else:
        # Fallback for older versions of torch_musa
        torch.cuda.is_bf16_supported = lambda: False

    # retain torch.empty reference
    original_empty = torch.empty
    # redifine torch.empty
    def patched_empty(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_empty(*args, **kwargs)
        return result
    torch.empty = patched_empty

patch_after_import_torch()