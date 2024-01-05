from typing import List

import torch

try:
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.optimizers import FusedAdam as Adam

    is_apex_available = True
except (ModuleNotFoundError, ImportError):
    is_apex_available = False
    Adam = torch.optim.Adam


def _get_params_for_partition_weight_optimization(modules, optimizer_overlap=False):
    """Divide params into groups.
    optimizer will swap each group from cpu to gpu when module.step() called by backward hook
    params_group: list of dict, store model params;
        0:  all not need `inject_optimizer` module's params
        [1..n]: other layer(s) params use by overlap offload optimizer step, will hold by overlap optimizer

    """
    params_group = [{"params": []}]  # {'params': []}
    wait_recurse_module = []
    overlap_modules = []
    for module in modules:
        # recurse to see: if isinstance FSDP and `inject_optimizer` equal False,
        # then put its params to one group,otherwise put into first
        params_group[0]["params"].extend([p for p in module.parameters(recurse=False)])
        for _, m in module.named_children():
            if hasattr(m, "inject_optimizer") and m.inject_optimizer:
                if optimizer_overlap:
                    overlap_modules.append(m)
                else:
                    params_group.append({"params": [p for p in m.parameters()]})
            else:
                # module not fit condition,to recurse
                wait_recurse_module.append(m)
    if wait_recurse_module:
        (
            sub_params_group,
            sub_overlap_modules,
        ) = _get_params_for_partition_weight_optimization(wait_recurse_module, optimizer_overlap)
        params_group[0]["params"].extend(sub_params_group[0]["params"])
        params_group.extend(sub_params_group[1:])
        overlap_modules.extend(sub_overlap_modules)
        # elif
    return params_group, overlap_modules


def register_overlap_optim(
    overlap_modules,
    build_optimizer_func,
    outer_optimizer,
    grad_scaler=None,
):
    """
    inner overlap optimizer must register on outer optimizer for save/restore
    call:
        module1.backward
        optimizer.step
    example:
        from atorch.optim.adam_offload import ( _get_params_for_partition_weight_optimization,
                                        register_overlap_optim,
                                        )
        param_groups, overlap_modules = _get_params_for_partition_weight_optimization(your_model, True)
        register_overlap_optim(
                overlap_modules,
                build_megatron_optimizer, # optimizer_func, will be called like: optimizer_func(param_groups, is_zero)
                optimizer,#optimizer instance, need implement add_overlap_optimizer for save and load_state_dict
                )
        return optimizer
    """
    stream_length = 2
    streams = [torch.cuda.Stream() for i in range(stream_length)]  # nested optimizer
    while hasattr(outer_optimizer, "optimizer"):
        outer_optimizer = outer_optimizer.optimizer
    for module in overlap_modules:
        optimizer = module.set_optimizer(build_optimizer_func, stream_length, streams)
        if not getattr(outer_optimizer, "add_overlap_optimizer", None):
            raise NotImplementedError(
                "optimizer %s need implement add_overlap_optimizer for save and restore" % (type(outer_optimizer))
            )
        outer_optimizer.add_overlap_optimizer(optimizer)


class PartitionAdam(Adam):
    """
    https://yuque.antfin-inc.com/ai-infra/atorch-doc/cd04ec#ouIM8
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stream_length = 2  # total 2 streams,h2d and d2h
        self.streams = [torch.cuda.Stream() for _ in range(self.stream_length)]
        self.inner_optimizers: List = []
        if not is_apex_available:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use PartitionAdam")

    def move_group_params(self, idx, group, stream, compute_device=None, non_blocking=False):
        # init: self.state = defaultdict(dict)
        optimizer_state = self.state[idx]
        # id(params)
        dtype = None
        if len(optimizer_state) == 0:
            # using one block to store optimizer state, and move between CPU/GPU
            numel = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )
                numel += p.data.numel()
                if dtype is None:
                    dtype = p.data.dtype
                else:
                    assert p.data.dtype == dtype, "Expect dtype should be equal,but %s != %s" % (
                        p.data.dtype,
                        dtype,
                    )
            # shape = (numel*2) ,one exp_avg,one exp_avg_sq,so there is only once copy
            optimizer_state["state_block"] = torch.zeros(numel * 2, device="cpu", dtype=dtype).pin_memory()
            optimizer_state["numel"] = numel
        if compute_device is None:
            compute_device = torch.cuda.current_device()
        # state_block = optimizer_state["state_block"].to(
        #     compute_device, non_blocking=non_blocking
        # )
        # state_block.record_stream(stream)
        return optimizer_state["state_block"], optimizer_state["numel"]

    def step(
        self,
        closure=None,
        grads=None,
        output_params=None,
        scale=None,
        grad_norms=None,
    ):
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated.  Simply initialize it "
                "identically to torch.optim.Adam, and call step() with no arguments."
            )
        loss = None
        if closure is not None:
            loss = closure()
        for i, group in enumerate(self.param_groups):
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1
            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            # removed prefetch
            compute_device = None
            h2d_stream = self.streams[0]
            d2h_stream = self.streams[-1]
            with torch.cuda.stream(h2d_stream):
                # copy state from cpu to gpu
                state_block_cpu, numel = self.move_group_params(i, group, h2d_stream)
                state_block = state_block_cpu.to(torch.cuda.current_device())
                # torch.cuda.current_stream().wait_stream(self.streams[i%self.stream_length])
                current_offset = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.data.is_sparse:
                        raise RuntimeError(
                            "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    if compute_device is None:
                        compute_device = p.device
                        assert compute_device != torch.device("cpu")
                    # save offset to state,so restore and restore can using
                    if p not in self.state:
                        self.state[p] = current_offset
                    else:
                        current_offset = self.state[p]
                    # exp_avg at [0: numel]

                    gpu_exp_avg = state_block[current_offset : current_offset + p.data.numel()].view(  # noqa: E203
                        p.data.shape
                    )
                    # exp_avg_sq at: [numel: 2*numel]
                    gpu_exp_avg_sq = state_block[
                        current_offset + numel : numel + current_offset + p.data.numel()  # noqa: E203
                    ].view(p.data.shape)
                    current_offset += p.data.numel()
                    if p.dtype == torch.float16:
                        g_16.append(p.grad.data)
                        p_16.append(p.data)
                        m_16.append(gpu_exp_avg)
                        v_16.append(gpu_exp_avg_sq)
                    elif p.dtype == torch.float32:
                        g_32.append(p.grad.data)
                        p_32.append(p.data)
                        m_32.append(gpu_exp_avg)
                        v_32.append(gpu_exp_avg_sq)
                    else:
                        raise RuntimeError("FusedAdam only support fp16 and fp32.")

                if len(g_16) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )
                if len(g_32) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )
            # do not sync and wait stream,will continue execute
            d2h_stream.wait_stream(h2d_stream)
            with torch.cuda.stream(d2h_stream):
                # copy state from gpu to cpu
                # cpu state will not release
                optimizer_state = self.state[i]
                optimizer_state["state_block"].copy_(state_block, non_blocking=True)
                state_block.record_stream(d2h_stream)
                # for p in group["params"]:
                #     p.grad = None

        return loss

    def add_overlap_optimizer(self, inner_optimizer):
        self.inner_optimizers.append(inner_optimizer)

    def state_dict(self):
        """pytorch default state_dict
        state will startwith 0 and store tensor enumerate idx,
        but our state have `state of offset` and  `state_block` of params
        if we have 1 param_groups and group have 2 params there will be: {0: state_block, 1: offset1,2:offset2}
        """
        for stream in self.streams:
            stream.synchronize()
        #
        # Save order indices instead of Tensors
        param_mappings = {}

        start_index = len(self.param_groups)  # HERE! should not startswith 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {id(p): i for i, p in enumerate(group["params"], start_index) if id(p) not in param_mappings}
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()}
        one_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }
        if self.inner_optimizers:
            one_dict["inner_optimizers"] = {}
            for idx, optimizer in enumerate(self.inner_optimizers):
                one_dict["inner_optimizers"][idx] = optimizer.state_dict()
        return one_dict

    def load_state_dict(self, state_dict):
        """pytorch optimizer default action:
        1. load optimizer state,and then move state to params device
            see: torch/optim/optimizer.py
            but we dont need move state to GPU,so move back

        """
        # can not use super
        ret = super().load_state_dict(state_dict)
        # we dont need move state to GPU,so move back
        for key in self.state:
            group_state = self.state[key]
            if isinstance(group_state, dict) and "state_block" in group_state:
                group_state["state_block"] = group_state["state_block"].to("cpu")
        if self.inner_optimizers and "inner_optimizers" in state_dict:
            for key, inner_state_dict in state_dict["inner_optimizers"].items():
                self.inner_optimizers[key].load_state_dict(inner_state_dict)

        return ret
