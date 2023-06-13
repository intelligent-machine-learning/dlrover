import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

try:
    from fmoe.gates import SwipeGate
    from fmoe.gates import SwitchGate as FastmoeSwitchGate
    from fmoe.layers import FMoE
    from fmoe.transformer import _Expert

    fastmoe_available = True
except (ImportError, ModuleNotFoundError):
    fastmoe_available = False
from atorch.common.log_utils import default_logger as logger


class MOEGroupContext(object):
    _moe_group = None
    _moe_group_ids = None
    _moe_ddp_group = None
    _moe_ddp_group_ids = None


def set_experts_process_group(this_rank, world_size, moe_group_size, rule="nearest"):
    """
    using this function to init process_group
    not holding process_group in Module object otherwise torch.save/load will cause exception
    params:
        this_rank:int
        world_size:int
        moe_group_size:int =4
        nearest:string [nearest, ] when it is nearest, using nearest as group; other: NotImplemented
    example:
        two group like: [0,1,2,3] [4,5,6,7]-> so that experts can be 4

    """
    assert world_size % moe_group_size == 0, "world_size:%d must be divide moe_group_size:%d" % (
        world_size,
        moe_group_size,
    )

    assert rule in ("nearest",)
    # all_ranks = list(range(world_size))
    expert_group_idx = this_rank // moe_group_size
    ddp_group_idx = this_rank % moe_group_size

    expert_data_parallel_size_ = world_size // moe_group_size
    # new data process group
    for i in range(expert_data_parallel_size_):  #
        ranks = list(range(moe_group_size * i, moe_group_size * (i + 1)))  # range(0,4),range(4,8)
        group = dist.new_group(ranks)
        if i == expert_group_idx:
            MOEGroupContext._moe_group = group
            MOEGroupContext._moe_group_ids = ranks
    # new moe process group
    for i in range(moe_group_size):  # 4
        ranks = list(range(i, world_size, moe_group_size))  # [0,4],[1,5],[2,6],[3,7]
        group = dist.new_group(ranks)
        if ddp_group_idx == i:
            MOEGroupContext._moe_ddp_group = group
            MOEGroupContext._moe_ddp_group_ids = ranks

    logger.info(
        "set_experts_process_group %d DP:%s experts:%s"
        % (this_rank, MOEGroupContext._moe_ddp_group_ids, MOEGroupContext._moe_group_ids)
    )


def get_experts_process_group():

    return MOEGroupContext._moe_group


def get_experts_ddp_process_group():
    return MOEGroupContext._moe_ddp_group


def get_experts_ddp_group_ids():
    return MOEGroupContext._moe_ddp_group_ids


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.group = group
        input = grad_output.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=ctx.group)
        return None, output


def _param_init_method(param, rng, sigma):
    r"""
    Init method based on N(0, sigma).
    Copied from Megatron-LM
    """
    device = param.device
    dtype = param.dtype
    weight = rng.normal(loc=0.0, scale=sigma, size=tuple(param.size()))
    param.data = torch.from_numpy(weight).to(dtype=dtype, device=device)


class Experts(Module):
    """Experts module in MoE layer.

    Args:
        num_local_experts (int): number of local experts
        model_dim (int): model's input hidden size
        ffn_dim (int): expert's ffn hidden size
        sigma (float, optional): parameters initial scale (default: 0.02)
        factor (int, optional): factor for out_experts' initial scale, usually
            number of transformer layers (default: 24)
    """

    def __init__(self, num_local_experts, model_dim, ffn_dim, sigma=0.02, factor=24):
        super(Experts, self).__init__()
        self.inner_experts = nn.Parameter(data=torch.ones(num_local_experts, model_dim, ffn_dim))
        self.out_experts = nn.Parameter(data=torch.ones(num_local_experts, ffn_dim, model_dim))
        self.activation_fn = F.relu
        self.num_local_experts = num_local_experts
        self.ffn_dim = ffn_dim

        # init params
        rng = np.random.default_rng(np.random.randint(2048) + torch.distributed.get_rank())
        _param_init_method(self.inner_experts, rng, sigma)
        std = sigma / math.sqrt(2.0 * factor)
        _param_init_method(self.out_experts, rng, std)

        for p in [self.inner_experts, self.out_experts]:
            p.expert = True  # type: ignore

    def forward(self, dispatched_input, outer_batch=False):
        intermediate = torch.einsum(
            "becm,emh->bech" if outer_batch else "ecm,emh->ech",
            dispatched_input,
            self.inner_experts,
        )
        assert intermediate.shape[-1] == self.ffn_dim
        activated_inters = self.activation_fn(intermediate)
        expert_output = torch.einsum(
            "bech,ehm->becm" if outer_batch else "ech,ehm->ecm",
            activated_inters,
            self.out_experts,
        )
        return expert_output


class MOELayer(Module):
    """MOELayer module which implements MixtureOfExperts as
       described in Gshard_. Prototype-format from M6-T.
    ::

        gate = SwitchGate(model_dim, num_experts)
        experts = Experts(num_local_experts, model_dim, ffn_dim)
        moe = MOELayer(gate, experts)
        output = moe(input)
        l_aux = moe.l_aux

    .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf
    .. _M6-T: https://arxiv.org/pdf/2105.15082.pdf

    Args:
        gate: gate network
        experts: experts network
        group (Optional): group for all-to-all communication (default: None)
        num_prototypes (Optional): numbers of prototypes (default: 1)
        fp16 (Optional): whether cast input to fp16 (default: False)
        outer_batch (Optional): Whether bring batch dim out of gate compute
            (default: False)
        dispatchV2 (Optional): Whether using idx-format moe dispatch and
            combine (default: False)
    """

    def __init__(
        self,
        gate,
        experts,
        group=None,
        num_prototypes=1,
        fp16=False,
        outer_batch=False,
        dispatchV2=False,
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.num_local_experts = self.experts.num_local_experts
        self.group = get_experts_process_group()  # FIXME: save / load
        self.world_size = dist.get_world_size(self.group)
        self.num_prototypes = num_prototypes
        self.fp16 = fp16
        self.outer_batch = outer_batch
        self.dispatchV2 = dispatchV2
        self.ffn_dim = self.experts.ffn_dim

    def __getstate__(self):
        # when save/load, module group will be ignore
        # using set_experts_process_group to set group
        attr_dict = super(MOELayer, self).__dict__
        attr_dict["group"] = None
        return attr_dict

    def __setstate__(self, state):
        state["group"] = get_experts_process_group()
        state["world_size"] = dist.get_world_size(state["group"])
        return super().__setstate__(state)

    def extra_repr(self) -> str:
        return f" world_size={self.world_size} \n"

    def forward(self, *input, **kwargs):
        if self.outer_batch:
            return self.forward_outer_batch(*input, **kwargs)
        else:
            return self.forward_ori(*input, **kwargs)

    def forward_ori(self, *input, **kwargs):

        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"

        # Implement Algorithm 2 from GShard paper.
        bs = input[0].shape[1]
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)  # shape: sm
        num_tokens = reshaped_input.shape[0]

        self.l_aux, combine_weights, dispatch_mask = self.gate(
            reshaped_input,
            bs,
            self.num_prototypes,
            self.outer_batch,
            self.dispatchV2,
        )

        reshaped_input = reshaped_input.unsqueeze(1).expand([num_tokens, self.num_prototypes, d_model])  # shape: szm

        if self.dispatchV2:
            weight, dispatch_idx_tuple = combine_weights, dispatch_mask
            dispatch_idx, num_experts, capacity = dispatch_idx_tuple
            # dispatch the inputs
            dispatched_input = self.idx_dispatch(dispatch_idx, reshaped_input, weight, num_experts, capacity)
        else:
            if self.fp16:
                dispatch_mask = dispatch_mask.half()
            else:
                dispatch_mask = dispatch_mask.float()
            # dispatch the inputs
            dispatched_input = torch.einsum("szfc,szm->zfcm", dispatch_mask, reshaped_input)

        dispatched_input = dispatched_input.reshape(-1, dispatched_input.shape[2], dispatched_input.shape[3])
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        # Re-shape after all-to-all: ecm -> ge'cm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)

        # chunk the inputs and pass through experts
        dispatched_input = dispatched_input.transpose(0, 1)
        assert dispatched_input.shape[0] == self.num_local_experts
        assert dispatched_input.shape[1] == self.world_size
        assert dispatched_input.shape[-1] == d_model
        dispatched_input = dispatched_input.reshape(self.num_local_experts, -1, d_model)

        # expert ffn
        expert_output = self.experts(dispatched_input)
        expert_output = expert_output.reshape(self.num_local_experts, self.world_size, -1, d_model).transpose(0, 1)

        expert_output = _AllToAll.apply(self.group, expert_output)

        # Re-shape back: ge'cm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        expert_output = expert_output.reshape(
            self.num_prototypes,
            (self.world_size * self.num_local_experts) // self.num_prototypes,
            -1,
            d_model,
        )

        # combine the output and bias
        if self.dispatchV2:
            combined_output = self.idx_combine(weight, dispatch_idx, expert_output, num_tokens)
        else:
            combined_output = torch.einsum("szfc,zfcm->sm", combine_weights, expert_output)
        combined_output = combined_output.reshape(input[0].shape)

        return combined_output

    def forward_outer_batch(self, *input, **kwargs):

        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"

        # Implement Algorithm 2 from GShard paper.
        bs = input[0].shape[1]
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        num_tokens = input[0].shape[0]  # num_tokens is seq if outer-batch
        reshaped_input = input[0].reshape(-1, d_model)  # shape: (sb)m

        self.l_aux, combine_weights, dispatch_mask = self.gate(
            reshaped_input,
            bs,
            self.num_prototypes,
            self.outer_batch,
            self.dispatchV2,
        )

        reshaped_input = reshaped_input.unsqueeze(1).expand(
            [num_tokens * bs, self.num_prototypes, d_model]
        )  # (sb)m -> (sb)zm
        reshaped_input = reshaped_input.reshape(-1, bs, *reshaped_input.shape[1:])  # (sb)zm -> sbzm

        if self.dispatchV2:
            weight, dispatch_idx_tuple = combine_weights, dispatch_mask
            dispatch_idx, num_experts, capacity = dispatch_idx_tuple
            # dispatch the inputs
            dispatched_input = self.idx_dispatch_outer_batch(
                dispatch_idx, reshaped_input, weight, num_experts, capacity
            )
        else:
            if self.fp16:
                dispatch_mask = dispatch_mask.half()
            else:
                dispatch_mask = dispatch_mask.float()
            # sbzec
            capacity = dispatch_mask.shape[-1]
            # dispatch the inputs
            dispatched_input = torch.einsum("sbzfc,sbzm->bzfcm", dispatch_mask, reshaped_input)

        dispatched_input = dispatched_input.reshape(
            bs,
            self.world_size * self.num_local_experts,
            dispatched_input.shape[-2],
            dispatched_input.shape[-1],
        )  # bzfcm -> becm
        dispatched_input = dispatched_input.transpose(0, 1)  # becm -> ebcm
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        # Re-shape after all-to-all: becm -> bge'cm
        dispatched_input = dispatched_input.reshape(bs, self.world_size, self.num_local_experts, capacity, d_model)

        # chunk the inputs and pass through experts
        dispatched_input = dispatched_input.transpose(1, 2)  # bge'cm -> be'gcm
        assert dispatched_input.shape[1] == self.num_local_experts
        assert dispatched_input.shape[2] == self.world_size
        assert dispatched_input.shape[-1] == d_model
        dispatched_input = dispatched_input.reshape(bs, self.num_local_experts, -1, d_model)  # be'gcm -> be'(gc)m

        # expert ffn
        expert_output = self.experts(dispatched_input, outer_batch=True)
        expert_output = expert_output.reshape(bs, self.num_local_experts, self.world_size, -1, d_model).transpose(
            1, 2
        )  # be'(gc)m -> bge'cm

        expert_output = expert_output.transpose(0, 1)  # bge'cm -> gbe'cm
        expert_output = _AllToAll.apply(self.group, expert_output)
        expert_output = expert_output.transpose(0, 1)  # gbe'cm -> bge'cm

        # Re-shape back: bge'cm -> becm
        expert_output = expert_output.reshape(bs, self.world_size * self.num_local_experts, -1, d_model)
        expert_output = expert_output.reshape(
            bs,
            self.num_prototypes,
            (self.world_size * self.num_local_experts) // self.num_prototypes,
            -1,
            d_model,
        )  # becm -> bzfcm

        # combine the output and bias
        if self.dispatchV2:
            combined_output = self.idx_combine_outer_batch(weight, dispatch_idx, expert_output, num_tokens)
        else:
            combined_output = torch.einsum("sbzfc,bzfcm->sbm", combine_weights, expert_output)  # sbzec,bzecm -> sbm
        combined_output = combined_output.reshape(input[0].shape)  # sbm -> sbm

        return combined_output

    def idx_dispatch(
        self,
        dispatch_idx,
        reshaped_input,
        weight,
        num_experts=None,
        capacity=None,
    ):
        """
        SwitchGate: dispatch_idx: sz2 (f, c), reshaped_input: szm,
                    weight: sz, dispatched_input: zfcm
        TopkGate: dispatch_idx: (ks)z2 (f, c), reshaped_input: szm,
                  weight: (ks)z, dispatched_input: zfcm
        """
        # compat TopkGate
        if reshaped_input.shape[0] != dispatch_idx.shape[0]:
            assert dispatch_idx.shape[0] % reshaped_input.shape[0] == 0
            reshaped_input = reshaped_input.repeat(int(dispatch_idx.shape[0] / reshaped_input.shape[0]), 1, 1)  # (ks)zm
        # e = zf, rectify e to f
        f = num_experts if num_experts else dispatch_idx[..., 0].max() + 1
        c = capacity if capacity else dispatch_idx[..., 1].max() + 1
        z, m = reshaped_input.shape[1], reshaped_input.shape[2]
        # Note(zy): Utilize extra place to filter idx where weight is zero,
        # (f*c+1)th. clone it, reserve dispatch_idx for combination
        dispatch_idx_cloned = dispatch_idx.clone()
        dispatch_idx_cloned[..., 0][weight == 0] = f
        dispatch_idx_cloned[..., 1][weight == 0] = 0
        dispatch_idx_flatten = dispatch_idx_cloned[..., 0] * c + dispatch_idx_cloned[..., 1]  # sz
        dispatch_idx_flatten = dispatch_idx_flatten.unsqueeze(-1).repeat(1, 1, m)  # szm
        dispatched_input = torch.zeros(
            [f * c + 1, z, m],
            device=reshaped_input.device,
            dtype=reshaped_input.dtype,
        )  # (fc+1)zm
        dispatched_input.scatter_(0, dispatch_idx_flatten, reshaped_input)  # core op
        dispatched_input = dispatched_input[:-1, ...]  # (fc)zm
        dispatched_input = dispatched_input.transpose(0, 1)  # z(fc)m
        dispatched_input = dispatched_input.reshape(z, f, c, m)  # zfcm
        return dispatched_input

    def idx_combine(self, weight, dispatch_idx, expert_output, num_tokens=None):
        """
        SwitchGate: weight: sz, dispatch_idx: sz2 (f, c),
                    expert_output: zfcm, combined_output: sm
        TopkGate: weight: (ks)z, dispatch_idx: (ks)z2 (f, c),
                  expert_output: zfcm, combined_output: sm
        """
        z, f, c, m = expert_output.shape
        dispatch_idx_flatten = dispatch_idx[..., 0] * c + dispatch_idx[..., 1]  # sz
        dispatch_idx_flatten = dispatch_idx_flatten.unsqueeze(-1).repeat(1, 1, m)  # szm
        expert_output = expert_output.permute(1, 2, 0, 3)  # fczm
        expert_output = expert_output.reshape(f * c, z, m)  # (fc)zm
        combined_output = torch.gather(expert_output, 0, dispatch_idx_flatten)  # szm
        # weight will filter zero weight idx's output
        combined_output = weight.unsqueeze(-1) * combined_output  # szm
        combined_output = combined_output.sum(dim=1)  # sm
        # (ks)m if topkgate
        if num_tokens and combined_output.shape[0] != num_tokens:
            combined_output = combined_output.reshape(-1, num_tokens, m)  # ksm
            combined_output = combined_output.sum(dim=0)  # sm
        return combined_output

    def idx_dispatch_outer_batch(
        self,
        dispatch_idx,
        reshaped_input,
        weight,
        num_experts=None,
        capacity=None,
    ):
        """
        dispatch_idx: sbz2 (f, c), reshaped_input: sbzm,
        weight: sbz, dispatched_input: bzfcm
        """
        f = num_experts if num_experts else dispatch_idx[..., 0].max() + 1
        c = capacity if capacity else dispatch_idx[..., 1].max() + 1
        b, z, m = (
            reshaped_input.shape[1],
            reshaped_input.shape[2],
            reshaped_input.shape[3],
        )
        # Note(zy): Utilize extra place to filter idx where weight is zero,
        # (f*c+1)th. clone it, reserve dispatch_idx for combination
        dispatch_idx_cloned = dispatch_idx.clone()
        dispatch_idx_cloned[..., 0][weight == 0] = f
        dispatch_idx_cloned[..., 1][weight == 0] = 0
        dispatch_idx_flatten = dispatch_idx_cloned[..., 0] * c + dispatch_idx_cloned[..., 1]  # sbz
        dispatch_idx_flatten = dispatch_idx_flatten.unsqueeze(-1).repeat(1, 1, 1, m)  # sbzm
        dispatched_input = torch.zeros(
            [f * c + 1, b, z, m],
            device=reshaped_input.device,
            dtype=reshaped_input.dtype,
        )  # (fc+1)bzm
        dispatched_input.scatter_(0, dispatch_idx_flatten, reshaped_input)  # core op
        dispatched_input = dispatched_input[:-1, ...]  # (fc)bzm
        dispatched_input = dispatched_input.permute(1, 2, 0, 3)  # bz(fc)m
        dispatched_input = dispatched_input.reshape(b, z, f, c, m)  # bzfcm
        return dispatched_input

    def idx_combine_outer_batch(self, weight, dispatch_idx, expert_output, num_tokens=None):
        """
        weight: sbz, dispatch_idx: sbz2 (f, c),
        expert_output: bzfcm, combined_output: sbm
        """
        b, z, f, c, m = expert_output.shape
        dispatch_idx_flatten = dispatch_idx[..., 0] * c + dispatch_idx[..., 1]  # sbz
        dispatch_idx_flatten = dispatch_idx_flatten.unsqueeze(-1).repeat(1, 1, 1, m)  # sbzm
        expert_output = expert_output.permute(2, 3, 0, 1, 4)  # fcbzm
        expert_output = expert_output.reshape(f * c, b, z, m)  # (fc)bzm
        combined_output = torch.gather(expert_output, 0, dispatch_idx_flatten)  # sbzm
        # weight will filter zero weight idx's output
        combined_output = weight.unsqueeze(-1) * combined_output  # sbzm
        combined_output = combined_output.sum(dim=2)  # sbm
        # (ks)m if topkgate
        if num_tokens and combined_output.shape[0] != num_tokens:
            raise NotImplementedError
        return combined_output


if fastmoe_available:

    class FMoETransformerMLP(FMoE):
        r"""
        A complete MoE MLP module in a Transformer block.
        * `activation` is the activation function to be used in MLP in each expert.
        * `d_hidden` is the dimension of the MLP layer.
        modify fastmoe FMoETransformerMLP, support torch.save/load
        """

        def __init__(
            self,
            num_expert=32,
            d_model=1024,
            d_hidden=4096,
            activation=torch.nn.GELU(),
            expert_dp_comm="none",
            expert_rank=0,
            **kwargs,
        ):
            super().__init__(num_expert=num_expert, d_model=d_model, **kwargs)
            self.experts = _Expert(num_expert, d_model, d_hidden, activation, rank=expert_rank)
            self.mark_parallel_comm(expert_dp_comm)

        def forward(self, inp: torch.Tensor):
            r"""
            This module wraps up the FMoE module with reshape, residual and layer
            normalization.
            """
            original_shape = inp.shape
            inp = inp.reshape(-1, self.d_model)
            output = super().forward(inp)
            return output.reshape(original_shape)

        def __getstate__(self):
            # TODO: when save/load, module moe_group will be ignore
            # using set_experts_process_group to set moe_group
            attr_dict = super(FMoE, self).__dict__
            attr_dict["moe_group"] = None
            attr_dict["slice_group"] = None
            return attr_dict

        def __setstate__(self, state):
            state.setdefault("moe_group", None)
            state.setdefault("slice_group", None)
            self.moe_group = state["moe_group"]  # or dist.GroupMember.WORLD
            self.slice_group = state["slice_group"]  # or dist.GroupMember.WORLD
            return super().__setstate__(state)

        def extra_repr(self) -> str:
            return (
                f"num_expert={self.num_expert} world_size={self.world_size} \n"
                f"d_model={self.d_model} experts_fused={self.experts_fused}"
            )


class MoEBertOutput(nn.Module):
    def __init__(self, config):
        """
        origin BertSelfOutput: dense->layernorm->dropout
        Moe: mlp->layernorm->dropout
        """
        super().__init__()
        self.config = config
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        expert_group = get_experts_process_group()  # FIXME: not support copy.deepcopy and torch.save/load
        world_size = expert_group.size() if expert_group else dist.get_world_size()
        if config.moe_impl == "atorch":
            from atorch.modules.moe import SwitchGate as AtorchSwitchGate
            from atorch.modules.moe import TopkGate

            if config.moe_gate == "switch":
                gate = AtorchSwitchGate(config.hidden_size, config.num_experts)
            else:
                gate = TopkGate(config.hidden_size, config.num_experts)
            num_local_experts = config.num_experts // world_size
            experts = Experts(num_local_experts, config.hidden_size, config.intermediate_size)
            self.mlp = MOELayer(gate, experts, outer_batch=config.outer_batch, dispatchV2=config.dispatchV2)
        else:
            if not fastmoe_available:
                raise ModuleNotFoundError("moe_impl=fastmoe need fastmoe and lib nccl-devel, pip install fastmoe first")
            if config.moe_gate == "switch":
                kwargs = {"gate": FastmoeSwitchGate, "top_k": 1}
            else:
                kwargs = {"gate": SwipeGate, "top_k": 2}
            num_local_experts = config.num_experts // world_size
            # expert: htoh4(in=d_model,out=d_hidden)-activation->h4toh(in=d_hidden,out=d_model)
            self.mlp = FMoETransformerMLP(  # num_expert have no (s)
                num_expert=num_local_experts,
                d_model=config.hidden_size,
                d_hidden=config.intermediate_size,
                world_size=world_size,
                moe_group=expert_group,
                **kwargs,
            )
            self.mlp.is_expert = True  # using in _ddp_params_and_buffers_to_ignore

    def forward(self, hidden_states):
        x = self.mlp(hidden_states)
        x = self.LayerNorm(x + hidden_states)
        x = self.dropout(x)
        return x
