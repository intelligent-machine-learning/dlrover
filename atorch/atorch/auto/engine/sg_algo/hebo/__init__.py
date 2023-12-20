import numpy as np

if not hasattr(np, "float"):
    setattr(np, "float", np.float32)
if not hasattr(np, "object"):
    setattr(np, "object", object)
