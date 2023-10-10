# As pipe stage model are wrapped in atorch, so we need another wrapper to handle save and load
import os
import pathlib

try:
    from pippy.SaveModule import CKPT_INDEX_JSON_FILENAME, _save_index, _save_optim_state, _save_params
except ImportError:
    try:
        # Try to import from the non-public path (older version of pippy)
        from pippy.hf._SaveModule import CKPT_INDEX_JSON_FILENAME, _save_index, _save_optim_state, _save_params
    except ImportError:

        def _save_index():
            raise NotImplementedError("save_checkpoint is not available in current version of the PiPPy.")

        def _save_params():
            raise NotImplementedError("save_checkpoint is not available in current version of the PiPPy.")

        def _save_optim_state():
            raise NotImplementedError("save_checkpoint is not available in current version of the PiPPy.")


try:
    from pippy.LoadModule import load_checkpoint
except ImportError:

    def load_checkpoint():
        raise NotImplementedError("load_checkpoint is not available in current version of the PiPPy.")


import torch.distributed as dist

from atorch.common.log_utils import default_logger as logger
from atorch.distributed import local_rank
from atorch.modules.distributed_modules.compilers.pipe_compiler.distributed_pippy_compiler import SafeStage
from atorch.modules.distributed_modules.compilers.pipe_compiler.StageInterleaver import InterleaverOptimizer

PIPE_CKPT = "pipe_ckpt"


def save_checkpoint(stage, checkpoint_dir="checkpoints", optimizer=None):
    """
    Save the entire model's(`stage`) metadata in an index file and the `submod`
    parameters in `checkpoint_dir`

    Args:
        stage(`Pipe`): model pipeline graph
        checkpoint_dir(`str`): directory where to save the index file and params binaries
                              defaults to `checkpoints`
        optimizer(`torch.optim.Optimizer`): optimizer whose state dict is to be saved
    """
    # create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # write index file in local rank 0
    if local_rank() == 0:
        _save_index(stage, checkpoint_dir=checkpoint_dir)

    _save_params(stage.submod, checkpoint_dir)  # type: ignore
    # save optimizer state, if passed
    if optimizer:
        _save_optim_state(optimizer, checkpoint_dir)  # type: ignore


# FIXME interleaved stages should have their separate checkpoint_dirs
def atorch_save_pipe_checkpoint(
    model,
    checkpoint_dir=PIPE_CKPT,
    optimizer=None,
):
    if not isinstance(model, SafeStage) or not isinstance(optimizer, InterleaverOptimizer):
        raise NotImplementedError(f"Save checkpoint is only implemented for SafeStage, not for {type(model)}")
    for stage, optim in zip(model.stage_interleaver.stages, optimizer.optimizers):
        save_checkpoint(stage, checkpoint_dir, optim)

    # barrier on all processes to wait for index file being properly saved on local rank 0
    dist.barrier()


def atorch_load_pipe_checkpoint(
    model,
    checkpoint_dir=PIPE_CKPT,
    index_filename=None,
    optim=None,
    device=None,
    dtype=None,
    checkpoint_prefix=None,
):
    if not isinstance(model, SafeStage) or not isinstance(optim, InterleaverOptimizer):
        raise NotImplementedError(f"Save checkpoint is only implemented for SafeStage, not for {type(model)}")

    if index_filename is None and checkpoint_dir is None:
        raise ValueError("Must specify either index_filename or checkpoint_dir")

    if index_filename is None:
        index_filename = os.path.join(checkpoint_dir, CKPT_INDEX_JSON_FILENAME)

    logger.info(f"Index file is loaded from {index_filename}")

    for i, stage in enumerate(model.stage_interleaver.stages):
        out = load_checkpoint(
            stage.submod,
            index_filename,
            optim=optim.optimizers[i],
            device=device,
            dtype=dtype,
            checkpoint_prefix=checkpoint_prefix,
        )
        if optim is not None:
            stage.submod, optim.optimizers[i] = out
        else:
            stage.submod = out
