"""Input/output checkpointing."""

import torch
from megatron import get_args
from megatron.checkpointing import load_checkpoint, save_checkpoint

from dlrover.python.common.singleton import singleton
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    MegatronCheckpointEngine,
)


@singleton
class DlroverCheckpointSaver(object):
    def __init__(self, checkpoint_dir):
        self.state_dict = {}
        self.path = ""
        self.engine = MegatronCheckpointEngine(checkpoint_dir)

    def save(self, state_dict, path):
        self.state_dict = state_dict
        self.path = path

    def load(self, path):
        state_dict = self.engine.load(resume_path=path)
        return state_dict


def save_checkpoint_to_storage(
    iteration, model, optimizer, opt_param_scheduler
):
    """
    Asynchronously save the the checkpointing state dict into the storage.
    The method will not wait for saving the checkpointing to the storage.
    """
    args = get_args()
    saver = DlroverCheckpointSaver(args.save)
    torch_save_func = torch.save
    torch.save = saver.save
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    saver.engine.save_to_storage(iteration, saver.state_dict, saver.path)
    torch.save = torch_save_func


def save_checkpoint_to_memory(
    iteration, model, optimizer, opt_param_scheduler
):
    """
    Synchronously save the the checkpointing state dict into the CPU memory.
    """
    args = get_args()
    saver = DlroverCheckpointSaver(args.save)
    torch_save_func = torch.save
    torch.save = saver.save
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    saver.engine.save_to_memory(iteration, saver.state_dict, saver.path)
    torch.save = torch_save_func


def load_checkpoint_(
    model, optimizer, opt_param_scheduler, load_arg="load", strict=True
):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    saver = DlroverCheckpointSaver(args.save)
    torch_load_func = torch.load
    torch.load = saver.load
    load_checkpoint(model, optimizer, opt_param_scheduler, load_arg, strict)
    torch.load = torch_load_func
