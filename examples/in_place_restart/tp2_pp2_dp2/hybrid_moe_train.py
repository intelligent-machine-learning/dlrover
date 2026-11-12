#!/usr/bin/env python3
# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eight-rank TP2 x PP2 x DP2 MoE recovery example.

EP=1 means there is no expert partition across additional expert-parallel
ranks. Stage 0 owns a TP-sharded embedding. Stage 1 owns TP-sharded MoE
experts and a TP-sharded output head. Each stage uses its own data-parallel
group.
"""

import argparse
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from recovery_protocol import (
    RecoveryCommand,
    atomic_write_json,
    fresh_reconfigure_uuid,
    load_recovery_command,
    read_json,
    wait_for_recovery_command,
)


WORLD_SIZE = 8
TP_SIZE = 2
PP_SIZE = 2
DP_SIZE = 2
EP_SIZE = 1

BATCH_SIZE = 8
SEQ_LEN = 32
VOCAB_SIZE = 1_000
EMBED_DIM = 128
EXPERT_HIDDEN_DIM = 512
NUM_EXPERTS = 4
TOP_K = 1
MODEL_SEED = 2026
DATA_SEED = 2027
CHECKPOINT_FORMAT_VERSION = 1
TOPOLOGY_ID = "tp2-pp2-dp2-ep1"


@dataclass(frozen=True)
class ParallelTopology:
    tp: int = TP_SIZE
    pp: int = PP_SIZE
    dp: int = DP_SIZE
    ep: int = EP_SIZE
    world_size: int = WORLD_SIZE

    def coordinates(self, rank: int) -> Tuple[int, int, int]:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"invalid global rank {rank}")
        tp_rank = rank % self.tp
        pp_rank = (rank // self.tp) % self.pp
        dp_rank = rank // (self.tp * self.pp)
        return tp_rank, pp_rank, dp_rank

    def rank(self, tp_rank: int, pp_rank: int, dp_rank: int) -> int:
        return (
            dp_rank * self.tp * self.pp
            + pp_rank * self.tp
            + tp_rank
        )

    def tp_groups(self) -> List[Tuple[int, ...]]:
        return [
            tuple(
                self.rank(tp_rank, pp_rank, dp_rank)
                for tp_rank in range(self.tp)
            )
            for dp_rank in range(self.dp)
            for pp_rank in range(self.pp)
        ]

    def pp_groups(self) -> List[Tuple[int, ...]]:
        return [
            tuple(
                self.rank(tp_rank, pp_rank, dp_rank)
                for pp_rank in range(self.pp)
            )
            for dp_rank in range(self.dp)
            for tp_rank in range(self.tp)
        ]

    def dp_groups(self) -> List[Tuple[int, ...]]:
        return [
            tuple(
                self.rank(tp_rank, pp_rank, dp_rank)
                for dp_rank in range(self.dp)
            )
            for pp_rank in range(self.pp)
            for tp_rank in range(self.tp)
        ]


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    category: str
    members: Tuple[int, ...]
    index: int


@dataclass
class HybridRuntime:
    rank: int
    local_rank: int
    device: torch.device
    topology: ParallelTopology
    tp_rank: int
    pp_rank: int
    dp_rank: int
    work_dir: Path
    store: Any
    default_group: Any
    group_specs: List[GroupSpec]
    local_groups: Dict[str, Any]
    stage_module: nn.Module
    optimizer: optim.Optimizer
    model: Optional[DDP] = None
    abort_lock: Any = field(
        default_factory=threading.Lock,
        repr=False,
    )
    aborted_epoch: int = 0
    global_step: int = 0
    epoch: int = 0
    batch_index: int = 0
    recovery_epoch: int = 0
    recovery_count: int = 0

    @property
    def tp_group(self) -> Any:
        return self._group("tp")

    @property
    def pp_group(self) -> Any:
        return self._group("pp")

    @property
    def dp_group(self) -> Any:
        return self._group("dp")

    def _group(self, category: str) -> Any:
        for group_id, group in self.local_groups.items():
            if group_id.startswith(f"{category}-"):
                return group
        raise RuntimeError(f"local {category} group is unavailable")


class RecoveryRequested(Exception):
    def __init__(self, command: RecoveryCommand) -> None:
        super().__init__(
            f"recovery epoch {command.recovery_epoch} requested"
        )
        self.command = command


class _AllReduceForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Any,
    ) -> torch.Tensor:
        del ctx
        output = tensor.contiguous().clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        del ctx
        return grad_output, None


class _AllGatherLastDim(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Any,
    ) -> torch.Tensor:
        tensor = tensor.contiguous()
        world_size = dist.get_world_size(group)
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=group)
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = dist.get_rank(group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        del ctx.group
        chunks = grad_output.contiguous().chunk(
            ctx.world_size,
            dim=-1,
        )
        return chunks[ctx.rank], None


class ParallelEmbeddingStage(nn.Module):
    def __init__(
        self,
        device: torch.device,
        tp_rank: int,
        tp_group: Any,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        full_weight = _seeded_tensor(
            (VOCAB_SIZE, EMBED_DIM),
            MODEL_SEED,
            scale=0.02,
        )
        local_width = EMBED_DIM // TP_SIZE
        start = tp_rank * local_width
        local_weight = full_weight[:, start : start + local_width]
        self.weight = nn.Parameter(local_weight.to(device))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        local_hidden = F.embedding(tokens, self.weight)
        return _AllGatherLastDim.apply(local_hidden, self.tp_group)


class TensorParallelExpert(nn.Module):
    def __init__(
        self,
        device: torch.device,
        tp_rank: int,
        seed: int,
    ) -> None:
        super().__init__()
        local_hidden = EXPERT_HIDDEN_DIM // TP_SIZE
        self.up = nn.Linear(
            EMBED_DIM,
            local_hidden,
            bias=False,
            device=device,
        )
        self.down = nn.Linear(
            local_hidden,
            EMBED_DIM,
            bias=False,
            device=device,
        )

        full_up = _seeded_tensor(
            (EXPERT_HIDDEN_DIM, EMBED_DIM),
            seed,
            scale=0.02,
        )
        start = tp_rank * local_hidden
        self.up.weight.data.copy_(
            full_up[start : start + local_hidden].to(device)
        )

        full_down = _seeded_tensor(
            (EMBED_DIM, EXPERT_HIDDEN_DIM),
            seed + 1,
            scale=0.02,
        )
        self.down.weight.data.copy_(
            full_down[:, start : start + local_hidden].to(device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.relu(self.up(x)))


class ParallelMoEStage(nn.Module):
    def __init__(
        self,
        device: torch.device,
        tp_rank: int,
        tp_group: Any,
    ) -> None:
        super().__init__()
        self.tp_group = tp_group
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K

        gate_weight = _seeded_tensor(
            (NUM_EXPERTS, EMBED_DIM),
            MODEL_SEED + 100,
            scale=0.02,
        )
        self.gate = nn.Linear(
            EMBED_DIM,
            NUM_EXPERTS,
            bias=False,
            device=device,
        )
        self.gate.weight.data.copy_(gate_weight.to(device))

        self.experts = nn.ModuleList(
            [
                TensorParallelExpert(
                    device,
                    tp_rank,
                    MODEL_SEED + 200 + expert_id * 10,
                )
                for expert_id in range(NUM_EXPERTS)
            ]
        )

        full_head = _seeded_tensor(
            (VOCAB_SIZE, EMBED_DIM),
            MODEL_SEED + 500,
            scale=0.02,
        )
        local_vocab = VOCAB_SIZE // TP_SIZE
        start = tp_rank * local_vocab
        self.head_weight = nn.Parameter(
            full_head[start : start + local_vocab].to(device)
        )
        self.head_bias = nn.Parameter(
            torch.zeros(local_vocab, device=device)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape
        flat_x = x.reshape(-1, embed_dim)
        probs = F.softmax(self.gate(flat_x), dim=-1)
        weights, selected = torch.topk(
            probs,
            self.top_k,
            dim=-1,
        )

        local_output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            selected_for_expert = selected == expert_id
            expert_weight = (
                weights * selected_for_expert
            ).sum(dim=-1, keepdim=True)
            mask = expert_weight.squeeze(-1) > 0
            if mask.any():
                local_output[mask] += (
                    expert(flat_x[mask]) * expert_weight[mask]
                )

        full_output = _AllReduceForward.apply(
            local_output,
            self.tp_group,
        )
        local_logits = F.linear(
            full_output,
            self.head_weight,
            self.head_bias,
        )
        logits = _AllGatherLastDim.apply(
            local_logits,
            self.tp_group,
        )
        logits = logits.reshape(batch_size, seq_len, VOCAB_SIZE)

        top1 = torch.argmax(probs, dim=-1)
        density = (
            torch.bincount(top1, minlength=NUM_EXPERTS).float()
            / top1.numel()
        )
        aux_metric = (
            (NUM_EXPERTS * density)
            * (NUM_EXPERTS * probs.mean(dim=0))
        ).mean()
        return logits, aux_metric.detach()


class LazyRandomDataset(Dataset):
    def __init__(
        self,
        sequence_length: int,
        length: int = 10_000,
        seed: int = DATA_SEED,
    ) -> None:
        self.sequence_length = sequence_length
        self.length = length
        self.seed = seed

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        tokens = torch.randint(
            0,
            VOCAB_SIZE,
            (self.sequence_length,),
            generator=generator,
        )
        return tokens, tokens

    def __len__(self) -> int:
        return self.length


def _seeded_tensor(
    shape: Tuple[int, ...],
    seed: int,
    scale: float,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator) * scale


def _log(rank: int, message: str) -> None:
    print(f"[rank {rank} pid {os.getpid()}] {message}", flush=True)


def _group_specs(topology: ParallelTopology) -> List[GroupSpec]:
    specs = [
        GroupSpec(
            group_id="default",
            category="default",
            members=tuple(range(topology.world_size)),
            index=0,
        )
    ]
    for category, groups in (
        ("tp", topology.tp_groups()),
        ("pp", topology.pp_groups()),
        ("dp", topology.dp_groups()),
    ):
        for index, members in enumerate(groups, start=1):
            specs.append(
                GroupSpec(
                    group_id=f"{category}-{index - 1}",
                    category=category,
                    members=members,
                    index=len(specs),
                )
            )
    return specs


def _get_default_group() -> Any:
    getter = getattr(dist, "_get_default_group", None)
    if getter is None:
        getter = dist.distributed_c10d._get_default_group
    return getter()


def _require_reconfigure_api() -> None:
    required = (
        "_supports_reconfigure",
        "_get_reconfigure_handle",
        "_reconfigure",
        "get_backend_impl",
    )
    missing = [name for name in required if not hasattr(dist, name)]
    if missing:
        raise RuntimeError(
            f"PyTorch reconfigure APIs are missing: {missing}"
        )
    if not dist.is_backend_available("nccl2"):
        raise RuntimeError("the nccl2 backend is unavailable")


def _new_reconfigurable_group(
    members: Tuple[int, ...],
    timeout: timedelta,
) -> Any:
    options = dist.ProcessGroupNCCL.Options()
    options.enable_reconfigure = True
    return dist.new_group(
        ranks=list(members),
        backend="nccl2",
        pg_options=options,
        sort_ranks=True,
        timeout=timeout,
    )


def _setup_distributed(
    args: argparse.Namespace,
) -> Tuple[
    int,
    int,
    torch.device,
    Any,
    Any,
    List[GroupSpec],
    Dict[str, Any],
]:
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise ValueError(
            f"this demo requires WORLD_SIZE={WORLD_SIZE}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    topology = ParallelTopology()
    work_dir = Path(args.work_dir)
    rendezvous_dir = work_dir / "rendezvous"
    rendezvous_dir.mkdir(parents=True, exist_ok=True)
    store = dist.FileStore(str(rendezvous_dir / "store"), WORLD_SIZE)

    _require_reconfigure_api()
    dist.init_process_group(
        backend="nccl2",
        store=store,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=args.dist_timeout_seconds),
        enable_reconfigure=True,
    )
    default_group = _get_default_group()

    specs = _group_specs(topology)
    local_groups: Dict[str, Any] = {}
    for spec in specs:
        if spec.category == "default":
            continue
        group = _new_reconfigurable_group(
            spec.members,
            timedelta(seconds=args.dist_timeout_seconds),
        )
        if rank in spec.members:
            local_groups[spec.group_id] = group

    local_groups["default"] = default_group
    return (
        rank,
        local_rank,
        device,
        store,
        default_group,
        specs,
        local_groups,
    )


def _store_barrier(
    store: Any,
    attempt_id: str,
    phase: str,
    rank: int,
    timeout_seconds: float,
) -> None:
    barrier_store = dist.PrefixStore(
        f"atorch-hybrid/{attempt_id}",
        store,
    )
    barrier_store.set(f"{phase}/done/{rank}", "1")
    keys = [
        f"{phase}/done/{peer}" for peer in range(WORLD_SIZE)
    ]
    barrier_store.wait(
        keys,
        timedelta(seconds=timeout_seconds),
    )


def _reconfigure_group(
    runtime: HybridRuntime,
    spec: GroupSpec,
    attempt_id: str,
    base_uuid: int,
    timeout_seconds: float,
) -> None:
    if runtime.rank not in spec.members:
        return
    group = runtime.local_groups[spec.group_id]
    if not dist._supports_reconfigure(group):
        raise RuntimeError(
            f"process group {spec.group_id} cannot be reconfigured"
        )

    group_store = dist.PrefixStore(
        f"atorch-hybrid/{attempt_id}/handles/{spec.group_id}",
        runtime.store,
    )
    group_store.set(
        f"rank/{runtime.rank}",
        dist._get_reconfigure_handle(group),
    )
    handle_keys = [
        f"rank/{member}" for member in spec.members
    ]
    group_store.wait(
        handle_keys,
        timedelta(seconds=timeout_seconds),
    )
    handles = [
        group_store.get(key).decode("utf-8")
        for key in handle_keys
    ]

    work = dist._reconfigure(
        uuid=base_uuid + spec.index,
        handles=handles,
        group=group,
        timeout=timedelta(seconds=timeout_seconds),
    )
    work.wait()
    _log(
        runtime.rank,
        f"reconfigured {spec.group_id} members={list(spec.members)}",
    )


def _reconfigure_topology(
    runtime: HybridRuntime,
    attempt_id: str,
    base_uuid: int,
    timeout_seconds: float,
) -> None:
    by_category: Dict[str, List[GroupSpec]] = {}
    for spec in runtime.group_specs:
        by_category.setdefault(spec.category, []).append(spec)

    for category in ("default", "tp", "pp", "dp"):
        for spec in by_category[category]:
            _reconfigure_group(
                runtime,
                spec,
                attempt_id,
                base_uuid,
                timeout_seconds,
            )
        _store_barrier(
            runtime.store,
            attempt_id,
            f"reconfigured/{category}",
            runtime.rank,
            timeout_seconds,
        )


def _validate_local_groups(runtime: HybridRuntime) -> None:
    default_tensor = torch.ones(1, device=runtime.device)
    dist.all_reduce(default_tensor)
    if default_tensor.item() != WORLD_SIZE:
        raise RuntimeError("default process group validation failed")

    tp_tensor = torch.ones(1, device=runtime.device)
    dist.all_reduce(tp_tensor, group=runtime.tp_group)
    if tp_tensor.item() != TP_SIZE:
        raise RuntimeError("TP process group validation failed")

    dp_tensor = torch.ones(1, device=runtime.device)
    dist.all_reduce(dp_tensor, group=runtime.dp_group)
    if dp_tensor.item() != DP_SIZE:
        raise RuntimeError("DP process group validation failed")

    pp_tensor = torch.ones(1, device=runtime.device)
    pp_rank = dist.get_rank(group=runtime.pp_group)
    if pp_rank == 0:
        dist.send(
            pp_tensor,
            group=runtime.pp_group,
            group_dst=1,
        )
    else:
        dist.recv(
            pp_tensor,
            group=runtime.pp_group,
            group_src=0,
        )
        if pp_tensor.item() != 1:
            raise RuntimeError("PP process group validation failed")

    _log(runtime.rank, "TP/PP/DP group validation passed")


def _bootstrap_topology(
    runtime: HybridRuntime,
    timeout_seconds: float,
) -> None:
    uuid_key = "atorch-hybrid/bootstrap-uuid"
    if runtime.rank == 0:
        runtime.store.set(uuid_key, str(fresh_reconfigure_uuid()))
    runtime.store.wait(
        [uuid_key],
        timedelta(seconds=timeout_seconds),
    )
    base_uuid = int(runtime.store.get(uuid_key).decode("utf-8"))
    _reconfigure_topology(
        runtime,
        "bootstrap",
        base_uuid,
        timeout_seconds,
    )
    _validate_local_groups(runtime)


def _find_local_group(
    local_groups: Dict[str, Any],
    category: str,
) -> Any:
    for group_id, group in local_groups.items():
        if group_id.startswith(f"{category}-"):
            return group
    raise RuntimeError(f"local {category} group is unavailable")


def _wrap_ddp(runtime: HybridRuntime) -> DDP:
    return DDP(
        runtime.stage_module,
        device_ids=[runtime.local_rank],
        output_device=runtime.local_rank,
        process_group=runtime.dp_group,
        find_unused_parameters=True,
    )


def _write_status(
    runtime: HybridRuntime,
    phase: str,
    **extra: Any,
) -> None:
    status = {
        "rank": runtime.rank,
        "pid": os.getpid(),
        "phase": phase,
        "global_step": runtime.global_step,
        "epoch": runtime.epoch,
        "batch_index": runtime.batch_index,
        "tp_rank": runtime.tp_rank,
        "pp_rank": runtime.pp_rank,
        "dp_rank": runtime.dp_rank,
        "recovery_epoch": runtime.recovery_epoch,
        "recovery_count": runtime.recovery_count,
        "timestamp_unix_ms": int(time.time() * 1000),
    }
    status.update(extra)
    atomic_write_json(
        runtime.work_dir
        / "status"
        / f"rank_{runtime.rank}.json",
        status,
    )


def _save_checkpoint(
    runtime: HybridRuntime,
    args: argparse.Namespace,
) -> None:
    checkpoint_root = runtime.work_dir / "checkpoints"
    checkpoint_dir = (
        checkpoint_root / f"checkpoint-{runtime.global_step:08d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize(runtime.local_rank)
    rank_path = checkpoint_dir / f"rank_{runtime.rank}.pt"
    temporary_path = rank_path.with_suffix(".pt.tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "topology_id": TOPOLOGY_ID,
        "rank": runtime.rank,
        "tp_rank": runtime.tp_rank,
        "pp_rank": runtime.pp_rank,
        "dp_rank": runtime.dp_rank,
        "model": runtime.stage_module.state_dict(),
        "optimizer": runtime.optimizer.state_dict(),
        "cursor": {
            "global_step": runtime.global_step,
            "epoch": runtime.epoch,
            "batch_index": runtime.batch_index,
        },
    }
    with temporary_path.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary_path), str(rank_path))

    barrier_id = (
        f"checkpoint-{runtime.recovery_epoch}-"
        f"{runtime.global_step}"
    )
    _store_barrier(
        runtime.store,
        barrier_id,
        "written",
        runtime.rank,
        args.recovery_timeout_seconds,
    )
    if runtime.rank == 0:
        manifest_path = checkpoint_root / "latest.json"
        manifest = {
            "checkpoint_id": f"hybrid-moe-step-{runtime.global_step}",
            "checkpoint_step": runtime.global_step,
            "checkpoint_path": str(manifest_path.resolve()),
            "topology_id": TOPOLOGY_ID,
            "world_size": WORLD_SIZE,
            "rank_paths": {
                str(rank): str(
                    (
                        checkpoint_dir / f"rank_{rank}.pt"
                    ).resolve()
                )
                for rank in range(WORLD_SIZE)
            },
        }
        atomic_write_json(manifest_path, manifest)
        _log(
            runtime.rank,
            f"committed hybrid checkpoint at step "
            f"{runtime.global_step}",
        )


def _load_checkpoint(
    runtime: HybridRuntime,
    command: RecoveryCommand,
) -> None:
    manifest = read_json(Path(command.checkpoint_path))
    if manifest is None:
        raise ValueError(
            "cannot read checkpoint manifest: "
            f"{command.checkpoint_path}"
        )
    if manifest.get("topology_id") != TOPOLOGY_ID:
        raise ValueError("checkpoint topology does not match")
    if manifest.get("world_size") != WORLD_SIZE:
        raise ValueError("checkpoint world size does not match")
    if manifest.get("checkpoint_step") != command.checkpoint_step:
        raise ValueError("checkpoint step does not match command")

    rank_path = Path(manifest["rank_paths"][str(runtime.rank)])
    checkpoint = torch.load(
        rank_path,
        map_location=runtime.device,
        weights_only=True,
    )
    if checkpoint.get("rank") != runtime.rank:
        raise ValueError("checkpoint rank does not match")
    if checkpoint.get("tp_rank") != runtime.tp_rank:
        raise ValueError("checkpoint TP rank does not match")
    if checkpoint.get("pp_rank") != runtime.pp_rank:
        raise ValueError("checkpoint PP rank does not match")
    if checkpoint.get("dp_rank") != runtime.dp_rank:
        raise ValueError("checkpoint DP rank does not match")

    cursor = checkpoint["cursor"]
    runtime.stage_module.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    runtime.optimizer.load_state_dict(checkpoint["optimizer"])
    runtime.global_step = cursor["global_step"]
    runtime.epoch = cursor["epoch"]
    runtime.batch_index = cursor["batch_index"]
    _log(
        runtime.rank,
        f"restored local shard at step {runtime.global_step}, "
        f"epoch {runtime.epoch}, batch {runtime.batch_index}",
    )


def _abort_local_groups(
    runtime: HybridRuntime,
    recovery_epoch: int = 0,
) -> None:
    with runtime.abort_lock:
        if (
            recovery_epoch > 0
            and recovery_epoch <= runtime.aborted_epoch
        ):
            return
        for group_id, group in runtime.local_groups.items():
            backend = dist.get_backend_impl(
                group=group,
                device=runtime.device,
            )
            try:
                backend.abort()
                _log(runtime.rank, f"revoked {group_id}")
            except Exception as error:
                _log(
                    runtime.rank,
                    f"failed to revoke {group_id}: {error}",
                )
        runtime.aborted_epoch = recovery_epoch


def _start_recovery_watcher(
    runtime: HybridRuntime,
    args: argparse.Namespace,
) -> Tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    command_path = Path(args.recovery_command)

    def watch() -> None:
        seen_epoch = runtime.recovery_epoch
        while not stop_event.wait(0.2):
            try:
                command = load_recovery_command(command_path)
            except (TypeError, ValueError):
                continue
            if (
                command is not None
                and command.recovery_epoch > seen_epoch
            ):
                _log(
                    runtime.rank,
                    f"control watcher observed recovery epoch "
                    f"{command.recovery_epoch}; revoking local groups",
                )
                _abort_local_groups(
                    runtime,
                    command.recovery_epoch,
                )
                seen_epoch = command.recovery_epoch

    thread = threading.Thread(
        target=watch,
        name=f"recovery-watcher-rank-{runtime.rank}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _detach_ddp(runtime: HybridRuntime) -> None:
    if runtime.model is None:
        return
    runtime.stage_module = runtime.model.module
    runtime.model = None
    gc.collect()


def _wait_for_all_ranks(
    runtime: HybridRuntime,
    attempt_id: str,
    timeout_seconds: float,
) -> None:
    ready_store = dist.PrefixStore(
        f"atorch-hybrid/{attempt_id}",
        runtime.store,
    )
    ready_store.set(f"ready/{runtime.rank}", "1")
    ready_keys = [
        f"ready/{rank}" for rank in range(WORLD_SIZE)
    ]
    ready_store.wait(
        ready_keys,
        timedelta(seconds=timeout_seconds),
    )


def _recover(
    runtime: HybridRuntime,
    args: argparse.Namespace,
    command: RecoveryCommand,
    replacement: bool,
) -> None:
    if command.world_size != WORLD_SIZE:
        raise ValueError("recovery world size does not match")
    if replacement and command.failed_rank != runtime.rank:
        raise ValueError("replacement rank does not match failure")
    remaining = command.deadline_unix_ms / 1000.0 - time.time()
    if remaining <= 0:
        raise TimeoutError("recovery deadline expired")
    timeout_seconds = min(
        args.recovery_timeout_seconds,
        remaining,
    )
    _log(
        runtime.rank,
        f"entering recovery epoch {command.recovery_epoch}; "
        f"replacement={replacement}",
    )

    _write_status(
        runtime,
        "quiescing",
        recovery_epoch=command.recovery_epoch,
    )
    if not replacement:
        _abort_local_groups(runtime, command.recovery_epoch)
    _detach_ddp(runtime)
    runtime.optimizer.zero_grad(set_to_none=True)
    _write_status(
        runtime,
        "quiesced",
        recovery_epoch=command.recovery_epoch,
    )
    _log(
        runtime.rank,
        "local training is quiesced; reconfiguring topology",
    )

    _reconfigure_topology(
        runtime,
        command.attempt_id,
        command.reconfigure_uuid,
        timeout_seconds,
    )
    _validate_local_groups(runtime)
    _write_status(
        runtime,
        "comm_ready",
        recovery_epoch=command.recovery_epoch,
    )

    _load_checkpoint(runtime, command)
    runtime.model = _wrap_ddp(runtime)
    runtime.model.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    runtime.recovery_epoch = command.recovery_epoch
    runtime.recovery_count += 1

    _wait_for_all_ranks(
        runtime,
        command.attempt_id,
        timeout_seconds,
    )
    dist.barrier()
    _write_status(runtime, "running")
    _log(
        runtime.rank,
        f"hybrid recovery complete at step {runtime.global_step}; "
        f"epoch={runtime.recovery_epoch}",
    )


def _make_loader(
    runtime: HybridRuntime,
    dataset: Dataset,
) -> DataLoader:
    sampler = DistributedSampler(
        dataset,
        num_replicas=DP_SIZE,
        rank=runtime.dp_rank,
        shuffle=True,
        seed=DATA_SEED,
    )
    sampler.set_epoch(runtime.epoch)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )


def _sync_tp_gradients(
    runtime: HybridRuntime,
    hidden: torch.Tensor,
) -> torch.Tensor:
    if runtime.pp_rank != 1:
        raise RuntimeError("only the PP last stage can sync gradients")
    if hidden.grad is None:
        raise RuntimeError("pipeline input gradient is missing")

    hidden_grad = hidden.grad.contiguous()
    dist.all_reduce(hidden_grad, group=runtime.tp_group)

    gate_grad = runtime.stage_module.gate.weight.grad
    if gate_grad is None:
        raise RuntimeError("MoE gate gradient is missing")
    dist.all_reduce(gate_grad, group=runtime.tp_group)
    return hidden_grad


def _pipeline_step(
    runtime: HybridRuntime,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert runtime.model is not None
    runtime.optimizer.zero_grad(set_to_none=True)

    if runtime.pp_rank == 0:
        hidden = runtime.model(inputs)
        dist.send(
            hidden.detach().contiguous(),
            group=runtime.pp_group,
            group_dst=1,
        )

        hidden_grad = torch.empty_like(hidden)
        with torch.no_grad():
            dist.recv(
                hidden_grad,
                group=runtime.pp_group,
                group_src=1,
            )
        hidden.backward(hidden_grad)
        runtime.optimizer.step()
        return None, None

    hidden = torch.empty(
        (BATCH_SIZE, SEQ_LEN, EMBED_DIM),
        device=runtime.device,
    )
    with torch.no_grad():
        dist.recv(
            hidden,
            group=runtime.pp_group,
            group_src=0,
        )
    hidden.requires_grad_(True)

    logits, aux_metric = runtime.model(hidden)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
    )
    loss.backward()
    hidden_grad = _sync_tp_gradients(runtime, hidden)
    dist.send(
        hidden_grad,
        group=runtime.pp_group,
        group_dst=0,
    )
    runtime.optimizer.step()
    return loss, aux_metric


def _is_comm_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "backend" in name
        or "nccl" in message
        or "collective" in message
        or "timed out" in message
        or "socket" in message
    )


def _run_training_segment(
    runtime: HybridRuntime,
    args: argparse.Namespace,
) -> None:
    dataset = LazyRandomDataset(SEQ_LEN)
    command_path = Path(args.recovery_command)
    interval_steps = 0
    interval_start = time.perf_counter()

    while runtime.global_step < args.steps:
        loader = _make_loader(runtime, dataset)
        batches_processed = 0
        for batch_index, (inputs, targets) in enumerate(loader):
            if runtime.global_step >= args.steps:
                break
            if batch_index < runtime.batch_index:
                continue

            command = load_recovery_command(command_path)
            if (
                command is not None
                and command.recovery_epoch > runtime.recovery_epoch
            ):
                raise RecoveryRequested(command)

            if (
                runtime.rank == args.fail_rank
                and runtime.global_step == args.fail_at_step
            ):
                _write_status(
                    runtime,
                    "injected_failure",
                    injected_failure=True,
                )
                _log(
                    runtime.rank,
                    f"injecting failure at step "
                    f"{runtime.global_step}",
                )
                os._exit(17)

            inputs = inputs.to(runtime.device)
            targets = targets.to(runtime.device)
            loss, aux_metric = _pipeline_step(
                runtime,
                inputs,
                targets,
            )

            runtime.global_step += 1
            runtime.batch_index = batch_index + 1
            if runtime.batch_index >= len(loader):
                runtime.epoch += 1
                runtime.batch_index = 0
            batches_processed += 1
            interval_steps += 1

            if (
                args.checkpoint_interval > 0
                and runtime.global_step
                % args.checkpoint_interval
                == 0
            ):
                _save_checkpoint(runtime, args)

            is_logger_rank = (
                runtime.pp_rank == 1
                and runtime.tp_rank == 0
                and runtime.dp_rank == 0
            )
            should_log = (
                interval_steps >= args.log_every
                or runtime.global_step >= args.steps
            )
            if is_logger_rank and should_log and loss is not None:
                torch.cuda.synchronize(runtime.local_rank)
                interval_seconds = time.perf_counter() - interval_start
                elapsed_ms = (
                    interval_seconds * 1000 / interval_steps
                )
                print(
                    f"iteration {runtime.global_step}/{args.steps} "
                    f"elapsed_ms={elapsed_ms:.3f} "
                    f"loss={loss.item():.6f} "
                    f"router_metric={float(aux_metric):.6f}",
                    flush=True,
                )
                interval_start = time.perf_counter()
                interval_steps = 0

            _write_status(runtime, "running")
            if runtime.global_step >= args.steps:
                break

        if runtime.global_step >= args.steps:
            break
        if batches_processed == 0:
            raise RuntimeError("dataloader made no progress")


def train(args: argparse.Namespace) -> None:
    (
        rank,
        local_rank,
        device,
        store,
        default_group,
        specs,
        local_groups,
    ) = _setup_distributed(args)
    topology = ParallelTopology()
    tp_rank, pp_rank, dp_rank = topology.coordinates(rank)

    tp_group = _find_local_group(local_groups, "tp")
    if pp_rank == 0:
        stage_module: nn.Module = ParallelEmbeddingStage(
            device,
            tp_rank,
            tp_group,
        )
    else:
        stage_module = ParallelMoEStage(
            device,
            tp_rank,
            tp_group,
        )
    optimizer = optim.Adam(stage_module.parameters(), lr=1e-3)
    runtime = HybridRuntime(
        rank=rank,
        local_rank=local_rank,
        device=device,
        topology=topology,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        work_dir=Path(args.work_dir),
        store=store,
        default_group=default_group,
        group_specs=specs,
        local_groups=local_groups,
        stage_module=stage_module,
        optimizer=optimizer,
    )
    _write_status(runtime, "initializing")
    watcher_stop = None
    watcher_thread = None

    try:
        if args.replacement:
            command = wait_for_recovery_command(
                Path(args.recovery_command),
                args.recovery_timeout_seconds,
            )
            _recover(runtime, args, command, replacement=True)
        else:
            _bootstrap_topology(
                runtime,
                args.recovery_timeout_seconds,
            )
            runtime.model = _wrap_ddp(runtime)
            runtime.model.train()
            _save_checkpoint(runtime, args)
            _write_status(runtime, "running")
            watcher_stop, watcher_thread = (
                _start_recovery_watcher(runtime, args)
            )

        while runtime.global_step < args.steps:
            try:
                _run_training_segment(runtime, args)
            except RecoveryRequested as error:
                _recover(
                    runtime,
                    args,
                    error.command,
                    replacement=False,
                )
            except Exception as error:
                if not _is_comm_error(error):
                    raise
                _log(
                    runtime.rank,
                    f"recoverable communication failure: {error}",
                )
                command = wait_for_recovery_command(
                    Path(args.recovery_command),
                    args.recovery_timeout_seconds,
                    minimum_epoch=runtime.recovery_epoch,
                )
                _recover(
                    runtime,
                    args,
                    command,
                    replacement=False,
                )

        dist.barrier()
        if (
            args.checkpoint_interval > 0
            and runtime.global_step % args.checkpoint_interval != 0
        ):
            _save_checkpoint(runtime, args)
        _write_status(runtime, "finished")
        _log(
            runtime.rank,
            f"finished {runtime.global_step} steps; "
            f"recoveries={runtime.recovery_count}",
        )
    except Exception as error:
        _write_status(runtime, "failed", error=repr(error))
        raise
    finally:
        if watcher_stop is not None:
            watcher_stop.set()
        if watcher_thread is not None:
            watcher_thread.join(timeout=2)
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--recovery-command", type=str, default="")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--dist-timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--recovery-timeout-seconds",
        type=float,
        default=240.0,
    )
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-at-step", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this demo requires eight CUDA devices")
    if torch.cuda.device_count() < WORLD_SIZE:
        raise SystemExit("this demo requires eight CUDA devices")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not args.recovery_command:
        args.recovery_command = str(
            work_dir / "recovery" / "command.json"
        )
    train(args)


if __name__ == "__main__":
    main()
