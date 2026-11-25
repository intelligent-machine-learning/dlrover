#!/usr/bin/env python3
# Copyright 2026 The DLRover Authors. All rights reserved.
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
"""Compatibility entry point for the two-node hybrid MoE example."""

import sys
from pathlib import Path

SHARED_EXAMPLE = (
    Path(__file__).resolve().parent.parent / "tp2_pp2_dp2"
)
if str(SHARED_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(SHARED_EXAMPLE))

from hybrid_moe_train import main  # noqa: E402


if __name__ == "__main__":
    main()
