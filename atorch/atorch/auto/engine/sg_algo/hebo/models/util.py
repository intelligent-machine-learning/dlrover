import numpy as np


def filter_nan(x, xe, y, keep_rule="any"):
    assert x is None or np.isfinite(x).all()
    assert xe is None or np.isfinite(xe).all()
    assert np.isfinite(y).any(), "No valid data in the dataset"

    if keep_rule == "any":
        valid_id = np.isfinite(y).any(axis=1)
    else:
        valid_id = np.isfinite(y).any(axis=1)
    x_filtered = x[valid_id] if x is not None else None
    xe_filtered = xe[valid_id] if xe is not None else None
    y_filtered = y[valid_id]
    return x_filtered, xe_filtered, y_filtered
