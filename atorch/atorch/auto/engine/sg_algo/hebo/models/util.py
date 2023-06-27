# Copyright 2022 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
