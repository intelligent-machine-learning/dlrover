# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input/output checkpointing."""

import os
import random
import sys

import numpy as np
import torch
from megatron import (
    get_args,
    mpu,
    print_rank_0,
    update_num_microbatches,
    utils,
)
from megatron.checkpointing import (
    check_checkpoint_args,
    fix_query_key_value_ordering,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_checkpoint_version,
    get_rng_state,
    read_metadata,
    set_checkpoint_version,
)

from dlrover.python.common.singleton import singleton
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    MegatronCheckpointEngine,
)


@singleton
class DlroverCheckpointSaver(object):
    def __init__(self, checkpoint_dir):
        self.engine = MegatronCheckpointEngine(checkpoint_dir)


def save_to_storage(iteration, model, optimizer, opt_param_scheduler):
    """
    Asynchronously save the the checkpointing state dict into the storage.
    The method will not wait for saving the checkpointing to the storage.
    """
    args = get_args()

    state_dict = get_checkpoint_state_dict(
        iteration, model, optimizer, opt_param_scheduler
    )
    # # Save.
    checkpoint_name = get_checkpoint_name(args.save, iteration)

    saver = DlroverCheckpointSaver(args.save)
    saver.engine.save_to_storage(iteration, state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        "  Notify the DLRover agent to save checkpoint to storage "
        "at iteration {:7d} to {}".format(iteration, args.save)
    )

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_to_memory(iteration, model, optimizer, opt_param_scheduler):
    """
    Synchronously save the the checkpointing state dict into the CPU memory.
    """
    args = get_args()

    state_dict = get_checkpoint_state_dict(
        iteration, model, optimizer, opt_param_scheduler
    )
    # Save.
    checkpoint_name = get_checkpoint_name(args.save, iteration)

    saver = DlroverCheckpointSaver(args.save)
    saver.engine.save_to_memory(iteration, state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        "  successfully saved checkpoint to memory "
        "at iteration {:7d} to {}".format(iteration, args.save)
    )

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_checkpoint_state_dict(
    iteration, model, optimizer, opt_param_scheduler
):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = utils.unwrap_model(model)

    print_rank_0(
        "saving checkpoint at iteration {:7d} to {}".format(
            iteration, args.save
        )
    )

    # collect rng state across data parallel ranks
    rng_state = get_rng_state()

    # Arguments, iteration, and model.
    state_dict = {}
    state_dict["args"] = args
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    if len(model) == 1:
        state_dict["model"] = model[0].state_dict_for_save_checkpoint()
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict["model%d" % i] = model[
                i
            ].state_dict_for_save_checkpoint()

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if opt_param_scheduler is not None:
            state_dict[
                "opt_param_scheduler"
            ] = opt_param_scheduler.state_dict()

    # RNG states.
    if not args.no_save_rng:
        state_dict["rng_state"] = rng_state
    return state_dict


def load_checkpoint(
    model, optimizer, opt_param_scheduler, load_arg="load", strict=True
):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    model = utils.unwrap_model(model)

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0(
            "WARNING: could not find the metadata file {} ".format(
                tracker_filename
            )
        )
        print_rank_0(
            "    will not load any checkpoints and will start from " "random"
        )
        return 0

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
    print_rank_0(
        f" loading checkpoint from {args.load} at iteration {iteration}"
    )

    # Load the checkpoint.
    try:
        saver = DlroverCheckpointSaver(args.save)
        state_dict = saver.load(checkpoint_name)
    except ModuleNotFoundError:
        # For backward compatibility.
        print_rank_0(" > deserializing using the old code structure ...")
        sys.modules["fp16.loss_scaler"] = sys.modules[
            "megatron.fp16_deprecated.loss_scaler"
        ]
        sys.modules["megatron.fp16.loss_scaler"] = sys.modules[
            "megatron.fp16_deprecated.loss_scaler"
        ]
        state_dict = torch.load(checkpoint_name, map_location="cpu")
        sys.modules.pop("fp16.loss_scaler", None)
        sys.modules.pop("megatron.fp16.loss_scaler", None)
    except BaseException as e:
        print_rank_0("could not load the checkpoint")
        print_rank_0(e)
        sys.exit()

    # set checkpoint version
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict["iteration"]
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict["total_iters"]
            except KeyError:
                print_rank_0(
                    "A metadata file exists but unable to load "
                    "iteration from checkpoint {}, exiting".format(
                        checkpoint_name
                    )
                )
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if "args" in state_dict:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(
            checkpoint_args, "consumed_train_samples", 0
        )
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(
            checkpoint_args, "consumed_valid_samples", 0
        )
    else:
        print_rank_0("could not find arguments in the checkpoint ...")

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(state_dict["model"], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict["model%d" % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(state_dict["optimizer"])
            if opt_param_scheduler is not None:
                if "lr_scheduler" in state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(
                        state_dict["lr_scheduler"]
                    )
                else:
                    opt_param_scheduler.load_state_dict(
                        state_dict["opt_param_scheduler"]
                    )
        except KeyError:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-optim or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if "rng_state" in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:

                    rng_state = state_dict["rng_state"][
                        mpu.get_data_parallel_rank()
                    ]
                else:
                    rng_state = state_dict["rng_state"][0]
                random.setstate(rng_state["random_rng_state"])
                np.random.set_state(rng_state["np_rng_state"])
                torch.set_rng_state(rng_state["torch_rng_state"])
                torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
                # Check for empty states array
                if not rng_state["rng_tracker_states"]:
                    raise KeyError
                mpu.get_cuda_rng_tracker().set_states(
                    rng_state["rng_tracker_states"]
                )
            else:  # backward compatability
                random.setstate(state_dict["random_rng_state"])
                np.random.set_state(state_dict["np_rng_state"])
                torch.set_rng_state(state_dict["torch_rng_state"])
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
                # Check for empty states array
                if not state_dict["rng_tracker_states"]:
                    raise KeyError
                mpu.get_cuda_rng_tracker().set_states(
                    state_dict["rng_tracker_states"]
                )
        except KeyError:
            print_rank_0(
                "Unable to load rng state from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the rng state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()

    # Some utilities want to load a checkpoint
    # without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {args.load} "
        f"at iteration {iteration}"
    )

    return iteration
