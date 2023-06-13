import warnings

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
    from apex.parallel import SyncBatchNorm
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Try using atorch LayerNorm but import fail:%s" % e)
    from torch.nn import LayerNorm as LayerNorm
    from torch.nn import SyncBatchNorm
