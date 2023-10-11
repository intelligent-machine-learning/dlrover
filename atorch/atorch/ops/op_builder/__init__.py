# Modifications Copyright 2023 AntGroups, Inc.
# ATorch Team

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import importlib
import os
import pkgutil
import sys

from .builder import OpBuilder, get_default_compute_capabilities

# Do not remove, required for abstract accelerator to detect if we have a atorch or 3p op_builder
__atorch__ = True

# List of all available op builders from atorch op_builder
try:
    import atorch.ops.op_builder  # noqa: F401 # type: ignore

    op_builder_dir = "atorch.ops.op_builder"
except ImportError:
    op_builder_dir = "op_builder"

__op_builders__ = []  # type: ignore

this_module = sys.modules[__name__]


def builder_closure(member_name):
    if op_builder_dir == "op_builder":
        # during installation time cannot get builder due to torch not installed,
        # return closure instead
        def _builder():
            from atorch.ops.accelerator import get_accelerator

            builder = get_accelerator().create_op_builder(member_name)
            return builder

        return _builder
    else:
        # during runtime, return op builder class directly
        from atorch.ops.accelerator import get_accelerator

        builder = get_accelerator().get_op_builder(member_name)
        return builder


# reflect builder names and add builder closure, such as 'TransformerBuilder()'
# creates op builder wrt current accelerator
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(this_module.__file__)]):  # type: ignore
    if module_name != "all_ops" and module_name != "builder":
        module = importlib.import_module(f".{module_name}", package=op_builder_dir)
        for member_name in module.__dir__():
            if member_name.endswith("Builder") and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = builder_closure(member_name)
