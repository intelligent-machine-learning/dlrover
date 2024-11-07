import os
import re
import shutil
from pathlib import Path
from typing import List

from atorch.common.log_utils import default_logger as logger

PREFIX_CHECKPOINT_DIR = "(checkpoint|iter)"
_re_checkpoint = re.compile(r"^(checkpoint|iter)(\-|\_)(\d+)$")


def get_last_checkpoint(folder):
    """
    support both megatron and transformer training framework

    Args:
        folder: train output_dir

    Returns:
        the subfolder path with the biggest suffix starting with "checkpoint"(hf format) or "iter"(megatron format).

    """

    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[2])))


def _sorted_checkpoints(
    output_dir, best_model_checkpoint=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob("[checkpoint iter]*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}(-|_)([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[2]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]  # type: ignore[misc]
    # Make sure we don't delete the best model.
    if best_model_checkpoint is not None and str(Path(best_model_checkpoint)) in checkpoints_sorted:
        best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))  # type: ignore[arg-type] # noqa: E501
        for i in range(best_model_index, len(checkpoints_sorted) - 2):
            checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
    return checkpoints_sorted  # type: ignore[return-value]


def _rotate_checkpoints(
    output_dir, save_total_limit: int = None, best_model_checkpoint: str = None, use_mtime=False
) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(
        use_mtime=use_mtime, output_dir=output_dir, best_model_checkpoint=best_model_checkpoint
    )
    if len(checkpoints_sorted) <= save_total_limit:
        return

    # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
    # we don't do to allow resuming.
    local_save_total_limit = save_total_limit
    if (
        best_model_checkpoint is not None
        and local_save_total_limit == 1
        and checkpoints_sorted[-1] != best_model_checkpoint
    ):
        local_save_total_limit = 2

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - local_save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")

        try:
            shutil.rmtree(checkpoint, ignore_errors=True)
        except FileNotFoundError:
            logger.warning(
                f"checkpoint {checkpoint} does not exit (maybe already get deleted). Please check whether "
                f"you are trying to delete the save checkpoint in multiple processes."
            )
