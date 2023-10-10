# Modifications Copyright 2023 AntGroups, Inc.
# ATorch Team

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Importing logger currently requires that torch is installed, hence the try...except
# TODO: Remove logger dependency on torch.
from atorch.common.log_utils import default_logger as accel_logger

try:
    from atorch.ops.accelerator.abstract_accelerator import BaseAccelerator as a1
except ImportError as e:
    accel_logger.error(f"{e}")
    a1 = None  # type: ignore
try:
    from atorch.ops.accelerator.abstract_accelerator import BaseAccelerator as a2
except ImportError as e:
    accel_logger.error(f"{e}")
    a2 = None  # type: ignore

accelerator = None


def _validate_accelerator(accel_obj):
    # because abstract_accelerator has different path during
    # build time (accelerator.abstract_accelerator)
    # and run time (atorch.accelerator.abstract_accelerator)
    # and extension would import the
    # run time abstract_accelerator/BaseAccelerator as its base
    # class, so we need to compare accel_obj with both base class.
    # if accel_obj is instance of BaseAccelerator in one of
    # accelerator.abstractor_accelerator
    # or atorch.accelerator.abstract_accelerator, consider accel_obj
    # is a conforming object
    if not ((a1 is not None and isinstance(accel_obj, a1)) or (a2 is not None and isinstance(accel_obj, a2))):
        raise AssertionError(f"{accel_obj.__class__.__name__} accelerator is not subclass of BaseAccelerator")

    # TODO: turn off is_available test since this breaks tests
    # assert accel_obj.is_available(), \
    #    f'{accel_obj.__class__.__name__} accelerator fails is_available() test'


def get_accelerator():
    global accelerator
    if accelerator is not None:
        return accelerator

    accelerator_name = "cuda"

    set_method = "auto detect"

    # 3. Set accelerator accordingly
    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDA_Accelerator

        accelerator = CUDA_Accelerator()

    _validate_accelerator(accelerator)
    if accel_logger is not None:
        accel_logger.info(f"Setting accelerator to {accelerator._name} ({set_method})")
    return accelerator


def set_accelerator(accel_obj):
    global accelerator
    _validate_accelerator(accel_obj)
    if accel_logger is not None:
        accel_logger.info(f"Setting accelerator to {accel_obj._name} (model specified)")
    accelerator = accel_obj
