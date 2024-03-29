# Modifications Copyright 2023 AntGroups, Inc.
# ATorch Team

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    #  This is populated by setup.py
    from atorch.ops.git_version_info_installed import (
        compatible_ops,
        git_branch,
        git_hash,
        installed_ops,
        torch_info,
        version,
    )
except ModuleNotFoundError:
    import os

    if os.path.isfile("version.txt"):
        # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
        version = open("version.txt", "r").read().strip()
    else:
        version = "0.0.0"
    git_hash = "[none]"
    git_branch = "[none]"

    from atorch.ops.op_builder.all_ops import ALL_OPS

    installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
    torch_info = {"version": "0.0", "cuda_version": "0.0", "hip_version": "0.0"}
