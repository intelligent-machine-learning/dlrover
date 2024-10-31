import torch.nn as nn

from atorch.kernels import cross_entropy_loss, npu_fuse_cross_entropy_loss
from atorch.utils.import_util import is_torch_npu_available


class AtorchCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        lse_square_scale=0.0,
        inplace_backward=False,
        process_group=None,
    ):
        """
        Arguments:
            ignored_index: int. If labels == ignored_index, the loss is set to 0.0.
            label_smoothing: float
            lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
                This is also referred to as "z-loss".
            inplace_backward: bool. If True, we do the backward pass in-place by modifying the logits.
                This saves memory.
            process_group: if not None, we're doing Tensor Parallel: each process is responsible for
            one part of the vocab. The loss will be aggregated across processes.
        """
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none' or 'sum'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        """
        Arguments:
            input: (batch, vocab_size)
            target: (batch,)
        Returns:
            losses: (batch,) if reduction is 'none', else (1,), dtype float
        """
        assert input.is_cuda and target.is_cuda, "Only support CUDA tensors"

        if is_torch_npu_available() and npu_fuse_cross_entropy_loss is not None:
            loss, _ = npu_fuse_cross_entropy_loss(
                input,
                target.int(),
                label_smoothing=self.label_smoothing,
                lse_square_scale=self.lse_square_scale,
                ignore_index=self.ignore_index,
            )
        else:
            loss = cross_entropy_loss(
                input,
                target,
                label_smoothing=self.label_smoothing,
                lse_square_scale=self.lse_square_scale,
                ignored_index=self.ignore_index,
                inplace_backward=self.inplace_backward,
                process_group=self.process_group,
            )
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
