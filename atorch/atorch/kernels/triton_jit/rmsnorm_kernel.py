import torch
import triton
import triton.language as tl
from triton import jit


@jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write  rstd
    tl.store(Rstd + row, rstd)
    # Normalize and multiply w
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y, mask=mask)


# backward: https://yuque.antfin-inc.com/ai-infra/atorch-design/nrzfivuloipx3o9l
@jit
def _rms_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    Lock,  # pointer to the lock
    stride,  # how much to increase the pointer when moving by 1 row
    weight_requires_grad,  # whether weight requires grad
    N,  # number of columns in X
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    # Offset locks and weights gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    origin_dy = tl.load(DY + cols, mask=mask, other=0)
    dy = origin_dy.to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask, other=0)
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    mean = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    if weight_requires_grad:
        # Accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(tl.float32)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        # First store doesn't accumulate
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        # Release the lock
        tl.atomic_xchg(Lock, 0)


@jit
def _rms_norm_bwd_dwdb(
    DW,  # pointer to the partial sum of weights gradient
    FINAL_DW,  # pointer to the weights gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


class AtorchRmsNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This rms norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _rms_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            rstd,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.weight_requires_grad = weight.requires_grad
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, v = ctx.saved_tensors
        weight_requires_grad = ctx.weight_requires_grad
        # heuristics for amount of parallel reduction stream for DW
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        # TODO: dtype=x.dtype will loss precision; but dtype=torch.float32 will be slow
        _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=torch.float32, device=w.device)

        ## need store fp32 to keep acc
        if weight_requires_grad:
            dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        _rms_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            _dw,
            x,
            w,
            v,
            locks,
            x_arg.stride(0),
            weight_requires_grad,
            N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps,
        )
        if weight_requires_grad:
            # accumulate partial sums in separate kernel
            _rms_norm_bwd_dwdb[grid](_dw, dw, GROUP_SIZE_M, N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
        else:
            dw = None
        return dx, dw, None


class AtorchRmsNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-06, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x):
        return AtorchRmsNormFunc.apply(x, self.weight, self.variance_epsilon)
