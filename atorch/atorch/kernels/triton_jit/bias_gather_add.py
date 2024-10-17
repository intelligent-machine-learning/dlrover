import torch

from atorch.utils.import_util import is_triton_available

if is_triton_available():
    import triton
    import triton.language as tl
else:
    from .triton_import_lib import Library

    triton = tl = Library


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _bias_gather_add_fw(
    inp,
    bias,
    out,
    bin_ids,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_id = tl.load(bin_ids + tl.program_id(0))

    inp += tl.multiple_of(token_id * NUM_COLUMNS, NUM_COLUMNS)
    out += tl.multiple_of(token_id * NUM_COLUMNS, NUM_COLUMNS)
    bias += tl.multiple_of(expert_id * NUM_COLUMNS, NUM_COLUMNS)

    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    for i in range(tl.cdiv(NUM_COLUMNS, BLOCK_SIZE)):
        mask = offsets < NUM_COLUMNS
        _inp = tl.load(inp + offsets, mask=mask)
        _bias = tl.load(bias + offsets, mask=mask)
        _inp += _bias
        tl.store(out + offsets, _inp, mask=mask)
        offsets += BLOCK_SIZE


def bias_gather_add_fw(inp, bias, bin_ids):
    assert inp.ndim == 2
    assert bias.ndim == 2
    assert bin_ids.ndim == 1
    assert inp.shape[1] == bias.shape[1]
    assert inp.shape[0] == bin_ids.shape[0]
    out = torch.empty_like(inp)

    _bias_gather_add_fw[(bin_ids.shape[0],)](
        inp,
        bias,
        out,
        bin_ids,
        NUM_COLUMNS=inp.shape[1],
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _bias_gather_add_bw(
    grad,
    bgrad,
    bin_ids,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_id = tl.load(bin_ids + tl.program_id(0))

    grad += tl.multiple_of(token_id * NUM_COLUMNS, NUM_COLUMNS)
    bgrad += tl.multiple_of(expert_id * NUM_COLUMNS, NUM_COLUMNS)

    offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    for i in range(tl.cdiv(NUM_COLUMNS, BLOCK_SIZE)):
        mask = offsets < NUM_COLUMNS
        _grad = tl.load(grad + offsets, mask=mask)
        tl.atomic_add(bgrad + offsets, _grad.to(tl.float32), mask=mask)
        offsets += BLOCK_SIZE


def create_bias_gather_add_bw():
    # TODO zy workaround: the first call to any new hidden size kernel seems to produce wrong result
    seen_hs = set()

    def bias_gather_add_bw(grad, bin_ids, num_experts):
        assert grad.ndim == 2
        assert grad.shape[0] == bin_ids.shape[0]
        bgrad = torch.zeros((num_experts, grad.shape[1]), device=grad.device, dtype=torch.float32)

        # TODO zy workaround
        nonlocal seen_hs
        if grad.shape[1] not in seen_hs:
            _bias_gather_add_bw[(bin_ids.shape[0],)](
                grad.detach().clone(),
                bgrad.detach().clone(),
                bin_ids.detach().clone(),
                NUM_COLUMNS=grad.shape[1],
            )
            seen_hs.add(grad.shape[1])

        _bias_gather_add_bw[(bin_ids.shape[0],)](
            grad,
            bgrad,
            bin_ids,
            NUM_COLUMNS=grad.shape[1],
        )
        return bgrad.to(grad.dtype)

    return bias_gather_add_bw


bias_gather_add_bw = create_bias_gather_add_bw()


class BiasGatherAddOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, bias, bin_ids):
        ctx.save_for_backward(bin_ids)
        ctx.num_experts = bias.shape[0]
        return bias_gather_add_fw(inp, bias, bin_ids)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        (bin_ids,) = ctx.saved_tensors
        bgrad = bias_gather_add_bw(grad, bin_ids, ctx.num_experts)
        return grad, bgrad, None


def bias_gather_add(inp: torch.Tensor, bias: torch.Tensor, bin_ids: torch.Tensor):
    """
    bias gather add fused by triton jit

    Arguments:
        inp: [token_num, hidden_size]
        bias: [expert_num, hidden_size]
        bin_ids: [token_num], element in [0, expert_num)
    Returns:
        torc.Tensor
    """
    return BiasGatherAddOp.apply(inp, bias, bin_ids)
