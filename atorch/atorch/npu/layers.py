import math

import torch

from atorch.kernels import npu_fusion_attention, npu_rms_norm
from atorch.utils.import_util import is_torch_npu_available


def create_additive_mask_by_breakpoint_mask(batch_size, seq_length, glm_mask):
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


def create_additive_pack_mask_by_startpoint_endpoint_mask(batch_size, seq_length, pack_glm_mask):
    assert pack_glm_mask.dim() == 3
    additive_mask_lst = []
    for bidx in range(batch_size):
        start_ends = pack_glm_mask
        starts = start_ends[bidx, 0, :]
        ends = start_ends[bidx, 1, :]

        # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
        m = torch.ones((seq_length, seq_length), device=pack_glm_mask.device).tril()
        for i in range(len(starts)):
            m[starts[i] :, : ends[i]] = 1

        additive_mask_lst.append(m.unsqueeze(0).unsqueeze(0))
    additive_mask_lst = torch.cat(additive_mask_lst, dim=0)

    return additive_mask_lst


def delete_minus_one_in_pack_glm_mask(pack_glm_mask, invalid_point=-1):
    if not (pack_glm_mask == invalid_point).any():
        return pack_glm_mask
    m = (pack_glm_mask == invalid_point).any(dim=0).to(torch.int32)
    first_neg_one_col_index = m.argmax()
    cleaned_tensor = pack_glm_mask[:, :first_neg_one_col_index]
    return cleaned_tensor


def npu_fa_supported_startpoint_endpoint_mask(q, k, v, glm_mask, kwargs):
    b, s_q, nh_q, hs = q.shape
    context_layer = torch.zeros([b, s_q, nh_q, hs], dtype=q.dtype, device=glm_mask.device)
    for i in range(b):
        cleaned_2d_pack_glm_mask = delete_minus_one_in_pack_glm_mask(glm_mask[i])
        additive_mask = create_additive_pack_mask_by_startpoint_endpoint_mask(
            1, s_q, cleaned_2d_pack_glm_mask.unsqueeze(0)
        )
        fa_output = npu_fusion_attention(
            q[i].unsqueeze(0).contiguous(),
            k[i].unsqueeze(0).contiguous(),
            v[i].unsqueeze(0).contiguous(),
            nh_q,
            "BSND",
            atten_mask=(1.0 - additive_mask).bool(),
            pse=None,
            padding_mask=None,
            scale=kwargs["scale"],
            keep_prob=kwargs["keep_prob"],
        )[0]
        context_layer[i] = torch.squeeze(fa_output, dim=0)
    return context_layer


def npu_fa_supported_breakpoint_mask(q, k, v, atten_mask, prefix, kwargs):
    _, _, nh_q, _ = q.shape
    kwargs["prefix"] = prefix
    kwargs["atten_mask"] = (1.0 - atten_mask).bool()
    kwargs["sparse_mode"] = 5
    kwargs["next_tockens"] = 0  # must set this value

    fa_output = npu_fusion_attention(q.contiguous(), k.contiguous(), v.contiguous(), nh_q, **kwargs)[0]
    return fa_output


def npu_fa_supported_single_query(q, k, v, kwargs):
    _, _, nh_q, _ = q.shape
    fa_output = npu_fusion_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        nh_q,
        kwargs["input_layout"],
        atten_mask=None,
        pse=None,
        padding_mask=None,
        scale=kwargs["scale"],
        keep_prob=kwargs["keep_prob"],
    )[0]
    return fa_output


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
    breakpoint_additive_mask=None,
    breakpoint_prefix=None,
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

    assert layout == "BSND", f"Only supprt BSND layout but got {layout}"
    supported_dtypes = (torch.float16, torch.bfloat16)
    assert q.dtype in supported_dtypes, f"The dtype of q should be {supported_dtypes} but got {q.dtype}"
    assert k.dtype in supported_dtypes, f"The dtype of k should be {supported_dtypes} but got {k.dtype}"
    assert v.dtype in supported_dtypes, f"The dtype of v should be {supported_dtypes} but got {v.dtype}"
    b_q, s_q, nh_q, hs = q.shape
    b_k, s_k, nh_k, hs = k.shape
    b_v, s_v, nh_v, hs = v.shape
    assert b_q == b_k and b_k == b_v, f"The batch size of query({b_q}), key({b_k}) and value({b_v}) must be identical."
    b = b_q
    assert b <= 2048, f"The batch size should smaller than 2048 but got {b}."

    if s_q != 1:
        assert s_q == s_k and s_q == s_v, (
            f"The sequence length of query({s_q}), key({s_k}) and value({s_v}) "
            "must be identical when the sequence length of query is not 1."
        )
    assert nh_k == nh_v, f"The head number of key({nh_k}) and value({nh_v}) must be identical."
    assert (
        nh_q % nh_k == 0
    ), f"The head number of query({nh_q}) must be divisible by that of key({nh_k}) or value({nh_v})."
    assert hs >= 1 and hs <= 512, f"head dimension must be larger equal than 1 and smaller equal than 512 but got {hs}"
    assert q.is_npu, f"q should on NPU but on {q.device}"
    assert k.is_npu, f"k should on NPU but on {k.device}"
    assert v.is_npu, f"v should on NPU but on {v.device}"

    kwargs = {
        "input_layout": layout,
        "pse": None,
        "padding_mask": None,
        "keep_prob": 1.0 - dropout_p,
        "gen_mask_parallel": gen_mask_parallel,
        "sync": sync,
        "scale": 1.0 / math.sqrt(hs) if softmax_scale is None else softmax_scale,
    }

    # single q
    if s_q == 1:
        return npu_fa_supported_single_query(q, k, v, kwargs)

    # Padding the sequence length to a multiple of 16.
    pad_dtype = q.dtype
    pad_device = q.device
    modulo = 16
    need_padding = False
    if s_q % modulo != 0:
        pad_seqlen = (s_q + modulo - 1) // modulo * modulo

        pad_q = torch.zeros([b, pad_seqlen, nh_q, hs], dtype=pad_dtype, device=pad_device)
        pad_k = torch.zeros([b, pad_seqlen, nh_k, hs], dtype=pad_dtype, device=pad_device)
        pad_v = torch.zeros([b, pad_seqlen, nh_v, hs], dtype=pad_dtype, device=pad_device)

        pad_q[:, :s_q, :, :] = q
        pad_k[:, :s_k, :, :] = k
        pad_v[:, :s_v, :, :] = v
        need_padding = True
    else:
        pad_q = q
        pad_k = k
        pad_v = v
        pad_seqlen = s_q

    assert pad_seqlen <= 524288, f"sequence length must smaller than 524288(512K) but got {pad_seqlen} after padding."

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
        fa_output = npu_fusion_attention(pad_q.contiguous(), pad_k.contiguous(), pad_v.contiguous(), nh_q, **kwargs)[0]
        context_layer = fa_output[:, :s_q, :, :]
        return context_layer

    if glm_mask is not None:
        assert isinstance(glm_mask, torch.Tensor), f"glm_mask shoule be a torch.Tensor but is a {type(glm_mask)}"
        assert glm_mask.is_npu, f"glm_mask should on NPU but on {glm_mask.device}"
        if glm_mask.dim() <= 1:
            # breakpoint-style mask
            atten_mask = (
                breakpoint_additive_mask
                if breakpoint_additive_mask is not None
                else create_additive_mask_by_breakpoint_mask(b, s_q, glm_mask)
            )
            if need_padding:
                pad_m = torch.ones([b, 1, pad_seqlen, pad_seqlen], dtype=pad_dtype, device=pad_device).tril()
                pad_m[:, :, :s_q, :s_q] = atten_mask
            else:
                pad_m = atten_mask
            if breakpoint_prefix is not None:
                prefix = breakpoint_prefix
            else:
                if glm_mask.dim() == 0:
                    prefix = [glm_mask.cpu().item()]
                else:
                    prefix = list(glm_mask.cpu().numpy())

            fa_output = npu_fa_supported_breakpoint_mask(pad_q, pad_k, pad_v, pad_m, prefix, kwargs)
            context_layer = fa_output[:, :s_q, :, :]
            return context_layer
        elif glm_mask.dim() == 3 and breakpoint_additive_mask is None:
            context_layer = npu_fa_supported_startpoint_endpoint_mask(pad_q, pad_k, pad_v, glm_mask, kwargs)
            return context_layer[:, :s_q, :, :]
        elif glm_mask.dim() == 4 or (glm_mask.dim() == 3 and breakpoint_additive_mask is not None):
            if glm_mask.dim() == 3 and breakpoint_additive_mask is not None:
                glm_mask = breakpoint_additive_mask
            if need_padding:
                pad_m = torch.ones([b, 1, pad_seqlen, pad_seqlen], dtype=pad_dtype, device=pad_device).tril()
                pad_m[:, :, :s_q, :s_q] = glm_mask
            else:
                pad_m = glm_mask
            fa_output = npu_fusion_attention(
                pad_q.contiguous(),
                pad_k.contiguous(),
                pad_v.contiguous(),
                nh_q,
                kwargs["input_layout"],
                atten_mask=(1.0 - pad_m).bool(),
                pse=None,
                padding_mask=None,
                scale=kwargs["scale"],
                keep_prob=kwargs["keep_prob"],
            )[0]
            context_layer = fa_output[:, :s_q, :, :]
            return context_layer
        else:
            raise ValueError(
                "Only support breakpoint-style glm mask or startpoint/endpoint-style glm mask, "
                f"but got {glm_mask} whose shape is {glm_mask.shape}"
            )


class AtorchNpuRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """RMS Normaliation module

        Arguments:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        return output
