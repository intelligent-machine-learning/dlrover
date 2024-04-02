import math

import torch

from atorch.utils.import_util import is_torch_npu_available


def create_additive_mask_by_glm_mask(batch_size, seq_length, glm_mask):
    is_scalar = torch.numel(glm_mask) == 1
    sep = glm_mask.item() if is_scalar else glm_mask
    # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
    m = torch.ones((1, seq_length, seq_length), device=glm_mask.device)
    mask = torch.arange(1, m.shape[-1] + 1, device=m.device).reshape(1, -1, 1)
    ids = torch.arange(1, m.shape[-1] + 1, device=m.device).reshape(1, 1, -1).expand(1, m.shape[-1], -1)
    m = (ids <= mask).type_as(m)

    if is_scalar:
        m[0, :, : int(sep)] = 1
    else:
        m = m.expand(batch_size, -1, -1)
        ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
        mask = ids < sep.view(-1, 1)
        m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
    m = m.unsqueeze(1)
    return m


def npu_fa_with_glm_mask(
    q,
    k,
    v,
    glm_mask=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    layout="BSND",
    gen_mask_parallel=True,
    sync=False,
):
    """
    npu_fa_with_glm_mask used on NPU that support glm mask and causal mask.
    dropout_p should be set to 0.0 during evaluation.
    q, k, v, mask should be half precision(torch.float16 or torch.bfloat16).

    We use the following notation:
        b: batch_size
        s_q, s_k: sequence length of Q and K
        nh: number of attention heads
        hs: head dimension

    Args:
        q: torch.Tensor with shape [b, s_q, nh, hs]
        k/v: torch.Tensor with shape [b, s_k, nh, hs]
        glm_mask: torch.Tensor with shape [batch_size] in torch.int64.
        dropout_p: Drop rate. Should be set to 0.0 during evaluation.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(hs).
        causal: When glm_mask is None, create a triangular mask based on the sequence length.
        layout: str. Only support BSND (batch_size, sequence_length, number of heads, head dimension).
        gen_mask_parallel: For debugging.
        sync: bool. For debugging.

    Returns:
        out:[b, s_q, nh, hs]

    ref:
    https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/apiref/fmkadptapi/ptaoplist_000437.html
    """
    if not is_torch_npu_available():
        raise RuntimeError("'npu_fa_with_glm_mask' could only be used on NPU")
    import torch_npu

    assert layout == "BSND", f"Only supprt BSND layout but got {layout}"
    supported_dtypes = (torch.float16, torch.bfloat16)
    assert q.dtype in supported_dtypes, f"The dtype of q should be {supported_dtypes} but got {q.dtype}"
    assert k.dtype in supported_dtypes, f"The dtype of k should be {supported_dtypes} but got {k.dtype}"
    assert v.dtype in supported_dtypes, f"The dtype of v should be {supported_dtypes} but got {v.dtype}"
    b, s_q, nh, hs = q.shape
    b, s_k, nh, hs = k.shape
    b, s_v, nh, hs = v.shape
    assert s_q == s_k and s_q == s_v, "The sequence length of query, key and value must be identical."
    assert hs in (64, 80, 96, 120, 128, 256), "head dimension must be 64, 80, 96, 120, 128, 256"
    assert q.is_npu, f"q should on NPU but on {q.device}"
    assert k.is_npu, f"k should on NPU but on {k.device}"
    assert v.is_npu, f"v should on NPU but on {v.device}"
    pad_dtype = q.dtype
    pad_device = q.device

    # Padding the sequence length to a multiple of 16.
    modulo = 16
    need_padding = False
    if s_q % modulo != 0:
        pad_seqlen = (s_q + modulo) // modulo * modulo

        pad_q = torch.zeros([b, pad_seqlen, nh, hs], dtype=pad_dtype, device=pad_device)
        pad_k = torch.zeros([b, pad_seqlen, nh, hs], dtype=pad_dtype, device=pad_device)
        pad_v = torch.zeros([b, pad_seqlen, nh, hs], dtype=pad_dtype, device=pad_device)

        pad_q[:, :s_q, :, :] = q
        pad_k[:, :s_k, :, :] = k
        pad_v[:, :s_v, :, :] = v
        need_padding = True
    else:
        pad_q = q
        pad_k = k
        pad_v = v
        pad_seqlen = s_q

    assert pad_seqlen <= 32768, f"sequence length must smaller than 32768(32K) but got {pad_seqlen} after padding."

    kwargs = {
        "input_layout": layout,
        "pse": None,
        "padding_mask": None,
        "keep_prob": 1.0 - dropout_p,
        "gen_mask_parallel": gen_mask_parallel,
        "sync": sync,
        "scale": 1.0 / math.sqrt(hs) if softmax_scale is None else softmax_scale,
    }

    if causal and glm_mask is None:
        atten_mask = torch.tril(torch.ones((b, s_q, s_q), device=pad_device)).view(b, 1, s_q, s_q)
        if need_padding:
            pad_m = torch.ones([b, 1, pad_seqlen, pad_seqlen], dtype=pad_dtype, device=pad_device).tril()
            pad_m[:, :, :s_q, :s_q] = atten_mask
        else:
            pad_m = atten_mask
        kwargs["atten_mask"] = (1.0 - pad_m).bool()
        kwargs["next_tockens"] = 0
        kwargs["sparse_mode"] = 0

    if glm_mask is not None:
        assert isinstance(glm_mask, torch.Tensor), f"glm_mask shoule be a torch.Tensor but is a {type(glm_mask)}"
        assert glm_mask.is_npu, f"glm_mask should on NPU but on {glm_mask.device}"
        glm_mask = glm_mask.to(torch.int64)
        atten_mask = create_additive_mask_by_glm_mask(b, s_q, glm_mask)
        if need_padding:
            pad_m = torch.ones([b, 1, pad_seqlen, pad_seqlen], dtype=pad_dtype, device=pad_device).tril()
            pad_m[:, :, :s_q, :s_q] = atten_mask
        else:
            pad_m = atten_mask
        if glm_mask.dim() == 0:
            prefix = [glm_mask.cpu().item()]
        else:
            prefix = list(glm_mask.cpu().numpy())
        kwargs["prefix"] = prefix
        kwargs["atten_mask"] = (1.0 - pad_m).bool()
        kwargs["sparse_mode"] = 5
        kwargs["next_tockens"] = 0  # must set this value

    fa_output = torch_npu.npu_fusion_attention(
        pad_q.contiguous(), pad_k.contiguous(), pad_v.contiguous(), nh, **kwargs
    )[0]
    context_layer = fa_output[:, :s_q, :, :]
    return context_layer
