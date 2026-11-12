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
"""Four-rank MoE DDP training with optional in-place rank recovery.

The MoE experts in this example are replicated on every rank. The only
distributed communication is DDP over the default process group. This keeps
the example focused on recovering that group without rebuilding the model or
optimizer on healthy ranks.
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    wait_for_recovery_command,
)


BATCH_SIZE = 32
SEQ_LEN = 64
VOCAB_SIZE = 1_000
EMBED_DIM = 128
NUM_EXPERTS = 4
TOP_K = 1
MODEL_SEED = 1234
DATA_SEED = 5678
CHECKPOINT_FORMAT_VERSION = 1
TOPOLOGY = "dp4-experts-replicated"


@dataclass(frozen=True)
class SlowPlan:
    rank: int
    mode: str
    delay_ms: float
    extra_cycles: int
    start_step: int
    end_step: int

    def active(self, rank: int, step: int) -> bool:
        return (
            self.rank >= 0
            and rank == self.rank
            and self.start_step <= step < self.end_step
        )


class RecoveryRequested(Exception):
    def __init__(self, command: RecoveryCommand) -> None:
        super().__init__(
            f"recovery epoch {command.recovery_epoch} requested"
        )
        self.command = command


@dataclass
class TrainingRuntime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    work_dir: Path
    store: Any
    process_group: Any
    dataset: "LazyRandomDataset"
    raw_model: nn.Module
    optimizer: optim.Optimizer
    model: Optional[DDP] = None
    global_step: int = 0
    epoch: int = 0
    batch_index: int = 0
    recovery_epoch: int = 0
    recovery_count: int = 0


class FeedForwardExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                FeedForwardExpert(input_dim, input_dim * 4)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = x.shape
        flat_x = x.reshape(-1, embed_dim)
        probs = F.softmax(self.gate(flat_x), dim=-1)
        weights, selected = torch.topk(probs, self.top_k, dim=-1)
        if self.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            selected_for_expert = selected == expert_id
            expert_weight = (weights * selected_for_expert).sum(
                dim=-1, keepdim=True
            )
            mask = expert_weight.squeeze(-1) > 0
            if mask.any():
                output[mask] += (
                    expert(flat_x[mask]) * expert_weight[mask]
                )

        top1 = torch.argmax(probs, dim=-1)
        density = (
            torch.bincount(top1, minlength=self.num_experts).float()
            / top1.numel()
        )
        aux_loss = (
            (self.num_experts * density)
            * (self.num_experts * probs.mean(dim=0))
        ).mean()
        return output.reshape(batch_size, seq_len, embed_dim), aux_loss


class ToyMoETransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.moe = MoELayer(EMBED_DIM, NUM_EXPERTS, TOP_K)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed(tokens)
        moe_output, aux_loss = self.moe(hidden)
        hidden = hidden + moe_output
        return self.head(self.norm(hidden)), aux_loss


class LazyRandomDataset(Dataset):
    """Deterministic synthetic data so recovery can reproduce a data cursor."""

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


def _log(rank: int, message: str) -> None:
    print(f"[rank {rank} pid {os.getpid()}] {message}", flush=True)


def _write_status(
    runtime: TrainingRuntime,
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
        "recovery_epoch": runtime.recovery_epoch,
        "recovery_count": runtime.recovery_count,
        "timestamp_unix_ms": int(time.time() * 1000),
    }
    status.update(extra)
    atomic_write_json(
        runtime.work_dir / "status" / f"rank_{runtime.rank}.json",
        status,
    )


def inject_slowdown(
    plan: SlowPlan,
    rank: int,
    step: int,
    device: int,
) -> None:
    if plan.rank < 0:
        return

    is_slow_rank = plan.active(rank, step)
    if plan.mode == "cpu-sleep":
        if is_slow_rank and plan.delay_ms:
            torch.cuda.synchronize(device)
            time.sleep(plan.delay_ms / 1_000.0)
        return

    if plan.mode != "cuda-sleep":
        return
    if not hasattr(torch.cuda, "_sleep"):
        raise RuntimeError(
            "this PyTorch build does not expose torch.cuda._sleep"
        )

    cycles = max(1, plan.extra_cycles if is_slow_rank else 1)
    torch.cuda._sleep(cycles)


def estimated_training_flops_per_token() -> int:
    gate_flops = 2 * EMBED_DIM * NUM_EXPERTS
    expert_hidden_dim = EMBED_DIM * 4
    expert_flops = 4 * EMBED_DIM * expert_hidden_dim
    head_flops = 2 * EMBED_DIM * VOCAB_SIZE
    forward_flops = gate_flops + expert_flops + head_flops
    return 3 * forward_flops


def print_performance_metrics(
    global_step: int,
    total_steps: int,
    loss_value: float,
    interval_steps: int,
    interval_seconds: float,
) -> None:
    interval_ms = interval_seconds * 1_000.0
    elapsed_ms_per_iteration = interval_ms / interval_steps
    tokens_per_gpu = interval_steps * BATCH_SIZE * SEQ_LEN
    tokens_per_second_per_gpu = tokens_per_gpu / max(
        1e-9, interval_seconds
    )
    tflops_per_gpu = (
        tokens_per_second_per_gpu
        * estimated_training_flops_per_token()
        / 1e12
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] "
        f"iteration {global_step}/{total_steps} "
        f"elapsed time per iteration (ms): "
        f"{elapsed_ms_per_iteration:.3f} | "
        f"throughput per GPU (TFLOP/s/GPU): {tflops_per_gpu:.6f} | "
        f"tokens/s/GPU: {tokens_per_second_per_gpu:.3f} | "
        f"loss: {loss_value:.6f}",
        flush=True,
    )


def _require_reconfigure_api() -> None:
    required = (
        "_get_reconfigure_handle",
        "_reconfigure",
        "_supports_reconfigure",
        "get_backend_impl",
    )
    missing = [name for name in required if not hasattr(dist, name)]
    if missing:
        raise RuntimeError(
            "in-place recovery requires PyTorch 2.14 or newer; "
            f"missing torch.distributed APIs: {missing}"
        )
    if not dist.is_backend_available("nccl2"):
        raise RuntimeError(
            "in-place recovery requires the experimental nccl2 backend"
        )


def _get_default_process_group() -> Any:
    getter = getattr(dist, "_get_default_group", None)
    if getter is None:
        getter = dist.distributed_c10d._get_default_group
    return getter()


def _setup_distributed(
    args: argparse.Namespace,
) -> Tuple[int, int, int, torch.device, Any, Any]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(
            "this example requires exactly four ranks; "
            f"torchrun reported WORLD_SIZE={world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this example requires four CUDA devices")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"requires {world_size} CUDA devices, found "
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    work_dir = Path(args.work_dir)
    rendezvous_dir = work_dir / "rendezvous"
    rendezvous_dir.mkdir(parents=True, exist_ok=True)
    store = dist.FileStore(str(rendezvous_dir / "store"), world_size)

    timeout = timedelta(seconds=args.dist_timeout_seconds)
    if args.backend == "nccl2":
        _require_reconfigure_api()
        dist.init_process_group(
            backend=args.backend,
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            enable_reconfigure=True,
        )
    else:
        dist.init_process_group(
            backend=args.backend,
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )

    return (
        rank,
        local_rank,
        world_size,
        device,
        store,
        _get_default_process_group(),
    )


def _reconfigure_group(
    store: Any,
    process_group: Any,
    rank: int,
    world_size: int,
    attempt_id: str,
    reconfigure_uuid: int,
    timeout_seconds: float,
) -> None:
    if not dist._supports_reconfigure(process_group):
        raise RuntimeError(
            "the default process group does not support reconfigure"
        )

    attempt_store = dist.PrefixStore(
        f"atorch-in-place/{attempt_id}", store
    )
    handle = dist._get_reconfigure_handle(process_group)
    attempt_store.set(f"handle/{rank}", handle)
    handle_keys = [
        f"handle/{handle_rank}" for handle_rank in range(world_size)
    ]
    attempt_store.wait(
        handle_keys,
        timedelta(seconds=timeout_seconds),
    )
    handles = [
        attempt_store.get(key).decode("utf-8") for key in handle_keys
    ]

    _log(
        rank,
        f"reconfiguring process group for {attempt_id}; "
        f"rank ordering={list(range(world_size))}",
    )
    work = dist._reconfigure(
        uuid=reconfigure_uuid,
        handles=handles,
        group=process_group,
        timeout=timedelta(seconds=timeout_seconds),
    )
    work.wait()


def _bootstrap_reconfigurable_group(
    args: argparse.Namespace,
    store: Any,
    process_group: Any,
    rank: int,
    world_size: int,
) -> None:
    uuid_key = "atorch-in-place/bootstrap-uuid"
    if rank == 0:
        store.set(uuid_key, str(fresh_reconfigure_uuid()))
    store.wait(
        [uuid_key],
        timedelta(seconds=args.recovery_timeout_seconds),
    )
    reconfigure_uuid = int(store.get(uuid_key).decode("utf-8"))

    _reconfigure_group(
        store=store,
        process_group=process_group,
        rank=rank,
        world_size=world_size,
        attempt_id="bootstrap",
        reconfigure_uuid=reconfigure_uuid,
        timeout_seconds=args.recovery_timeout_seconds,
    )
    tensor = torch.ones(1, device=torch.cuda.current_device())
    dist.all_reduce(tensor)
    if tensor.item() != world_size:
        raise RuntimeError("initial process-group validation failed")


def _build_model(
    device: torch.device,
) -> Tuple[nn.Module, optim.Optimizer]:
    torch.manual_seed(MODEL_SEED)
    model = ToyMoETransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def _wrap_ddp(
    raw_model: nn.Module,
    local_rank: int,
) -> DDP:
    return DDP(
        raw_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def _checkpoint_payload(
    runtime: TrainingRuntime,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_step": runtime.global_step,
        "model": runtime.raw_model.state_dict(),
        "optimizer": runtime.optimizer.state_dict(),
        "cursor": {
            "global_step": runtime.global_step,
            "epoch": runtime.epoch,
            "batch_index": runtime.batch_index,
        },
        "topology": TOPOLOGY,
        "world_size": runtime.world_size,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "steps": args.steps,
        },
    }


def _save_checkpoint(
    runtime: TrainingRuntime,
    args: argparse.Namespace,
) -> Optional[Path]:
    if runtime.rank != 0:
        return None

    checkpoint_dir = runtime.work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir / f"checkpoint-{runtime.global_step:08d}.pt"
    )
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")

    torch.cuda.synchronize(runtime.local_rank)
    payload = _checkpoint_payload(runtime, args)
    with temporary_path.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary_path), str(checkpoint_path))

    manifest = {
        "checkpoint_id": f"moe-step-{runtime.global_step}",
        "checkpoint_step": runtime.global_step,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "committed_unix_ms": int(time.time() * 1000),
        "topology": TOPOLOGY,
        "world_size": runtime.world_size,
    }
    atomic_write_json(checkpoint_dir / "latest.json", manifest)
    _log(runtime.rank, f"committed checkpoint at step {runtime.global_step}")
    return checkpoint_path


def _load_checkpoint(
    runtime: TrainingRuntime,
    command: RecoveryCommand,
) -> None:
    checkpoint_path = Path(command.checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=runtime.device,
        weights_only=True,
    )

    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if checkpoint.get("topology") != TOPOLOGY:
        raise ValueError("checkpoint topology does not match this job")
    if checkpoint.get("world_size") != runtime.world_size:
        raise ValueError("checkpoint world size does not match this job")

    cursor = checkpoint["cursor"]
    if cursor["global_step"] != command.checkpoint_step:
        raise ValueError(
            "recovery command checkpoint step does not match checkpoint"
        )

    runtime.raw_model.load_state_dict(checkpoint["model"], strict=True)
    runtime.optimizer.load_state_dict(checkpoint["optimizer"])
    runtime.global_step = cursor["global_step"]
    runtime.epoch = cursor["epoch"]
    runtime.batch_index = cursor["batch_index"]
    _log(
        runtime.rank,
        f"restored checkpoint step {runtime.global_step}, "
        f"epoch {runtime.epoch}, batch {runtime.batch_index}",
    )


def _make_loader(
    runtime: TrainingRuntime,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        runtime.dataset,
        num_replicas=runtime.world_size,
        rank=runtime.rank,
        shuffle=True,
        seed=DATA_SEED,
    )
    sampler.set_epoch(runtime.epoch)
    loader = DataLoader(
        runtime.dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    return loader, sampler


def _abort_reconfigurable_group(runtime: TrainingRuntime) -> None:
    backend = dist.get_backend_impl(
        group=runtime.process_group,
        device=runtime.device,
    )
    # In nccl2 reconfigure mode, abort revokes in-flight work and keeps the
    # process-group shell available for reconfigure.
    backend.abort()


def _detach_ddp(runtime: TrainingRuntime) -> None:
    if runtime.model is None:
        return
    runtime.raw_model = runtime.model.module
    runtime.model = None
    gc.collect()


def _wait_for_all_ranks(
    runtime: TrainingRuntime,
    attempt_id: str,
    timeout_seconds: float,
) -> None:
    ready_store = dist.PrefixStore(
        f"atorch-in-place/{attempt_id}", runtime.store
    )
    ready_store.set(f"ready/{runtime.rank}", "1")
    ready_keys = [
        f"ready/{ready_rank}" for ready_rank in range(runtime.world_size)
    ]
    ready_store.wait(
        ready_keys,
        timedelta(seconds=timeout_seconds),
    )


def _recover(
    runtime: TrainingRuntime,
    args: argparse.Namespace,
    command: RecoveryCommand,
    replacement: bool,
) -> None:
    if args.backend != "nccl2":
        raise RuntimeError(
            "in-place recovery requires --backend nccl2"
        )
    if command.world_size != runtime.world_size:
        raise ValueError("recovery command world size does not match")
    if replacement and command.failed_rank != runtime.rank:
        raise ValueError(
            "replacement process does not match the failed logical rank"
        )
    remaining_seconds = (
        command.deadline_unix_ms / 1000.0 - time.time()
    )
    if remaining_seconds <= 0:
        raise TimeoutError(
            f"recovery epoch {command.recovery_epoch} expired"
        )
    recovery_timeout = min(
        args.recovery_timeout_seconds,
        remaining_seconds,
    )

    _write_status(
        runtime,
        "quiescing",
        recovery_epoch=command.recovery_epoch,
    )
    if not replacement:
        _abort_reconfigurable_group(runtime)
    _detach_ddp(runtime)
    runtime.optimizer.zero_grad(set_to_none=True)
    _write_status(
        runtime,
        "quiesced",
        recovery_epoch=command.recovery_epoch,
    )

    _reconfigure_group(
        store=runtime.store,
        process_group=runtime.process_group,
        rank=runtime.rank,
        world_size=runtime.world_size,
        attempt_id=command.attempt_id,
        reconfigure_uuid=command.reconfigure_uuid,
        timeout_seconds=recovery_timeout,
    )
    validation = torch.ones(1, device=runtime.device)
    dist.all_reduce(validation)
    if validation.item() != runtime.world_size:
        raise RuntimeError("reconfigured process-group validation failed")
    _write_status(
        runtime,
        "comm_ready",
        recovery_epoch=command.recovery_epoch,
    )

    _load_checkpoint(runtime, command)
    runtime.model = _wrap_ddp(runtime.raw_model, runtime.local_rank)
    runtime.model.train()
    runtime.optimizer.zero_grad(set_to_none=True)
    runtime.recovery_epoch = command.recovery_epoch
    runtime.recovery_count += 1

    _wait_for_all_ranks(
        runtime,
        command.attempt_id,
        recovery_timeout,
    )
    dist.barrier()
    _write_status(runtime, "running")
    _log(
        runtime.rank,
        f"in-place recovery complete at step {runtime.global_step}; "
        f"recovery epoch={runtime.recovery_epoch}",
    )


def _is_comm_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "backend" in name
        or "nccl" in message
        or "collective" in message
        or "timed out" in message
    )


def _run_training_segment(
    runtime: TrainingRuntime,
    args: argparse.Namespace,
) -> None:
    assert runtime.model is not None
    criterion = nn.CrossEntropyLoss()
    plan = SlowPlan(
        rank=args.slow_rank,
        mode=args.slow_mode,
        delay_ms=args.slow_delay_ms,
        extra_cycles=args.slow_extra_cycles,
        start_step=args.slow_start_step,
        end_step=args.slow_end_step,
    )
    recovery_command_path = Path(args.recovery_command)

    interval_steps = 0
    interval_start = time.perf_counter()

    while runtime.global_step < args.steps:
        loader, _ = _make_loader(runtime)
        batches_processed = 0

        for batch_index, (inputs, targets) in enumerate(loader):
            if runtime.global_step >= args.steps:
                break
            if batch_index < runtime.batch_index:
                continue

            command = load_recovery_command(recovery_command_path)
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
                    f"injecting process failure at step "
                    f"{runtime.global_step}",
                )
                os._exit(17)

            inputs = inputs.to(runtime.device, non_blocking=True)
            targets = targets.to(runtime.device, non_blocking=True)
            inject_slowdown(
                plan,
                runtime.rank,
                runtime.global_step,
                runtime.local_rank,
            )

            runtime.optimizer.zero_grad(set_to_none=True)
            logits, aux_loss = runtime.model(inputs)
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
            )
            (loss + 0.01 * aux_loss).backward()
            runtime.optimizer.step()

            runtime.global_step += 1
            runtime.batch_index = batch_index + 1
            if runtime.batch_index >= len(loader):
                runtime.epoch += 1
                runtime.batch_index = 0
            batches_processed += 1
            interval_steps += 1

            if (
                args.checkpoint_interval > 0
                and runtime.global_step % args.checkpoint_interval == 0
            ):
                _save_checkpoint(runtime, args)

            should_log = (
                interval_steps >= args.log_every
                or runtime.global_step >= args.steps
            )
            if runtime.rank == 0 and should_log:
                torch.cuda.synchronize(runtime.local_rank)
                interval_seconds = time.perf_counter() - interval_start
                print_performance_metrics(
                    global_step=runtime.global_step,
                    total_steps=args.steps,
                    loss_value=loss.item(),
                    interval_steps=interval_steps,
                    interval_seconds=interval_seconds,
                )
                interval_start = time.perf_counter()
                interval_steps = 0

            _write_status(runtime, "running")

            if runtime.global_step >= args.steps:
                break

        if runtime.global_step >= args.steps:
            break
        if batches_processed == 0:
            raise RuntimeError(
                "dataloader made no progress while resuming"
            )


def train(args: argparse.Namespace) -> None:
    (
        rank,
        local_rank,
        world_size,
        device,
        store,
        process_group,
    ) = _setup_distributed(args)

    if args.backend == "nccl2" and not args.replacement:
        _bootstrap_reconfigurable_group(
            args,
            store,
            process_group,
            rank,
            world_size,
        )

    dataset = LazyRandomDataset(SEQ_LEN)
    raw_model, optimizer = _build_model(device)
    runtime = TrainingRuntime(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        work_dir=Path(args.work_dir),
        store=store,
        process_group=process_group,
        dataset=dataset,
        raw_model=raw_model,
        optimizer=optimizer,
    )
    _write_status(runtime, "initializing")

    try:
        if args.replacement:
            command = wait_for_recovery_command(
                Path(args.recovery_command),
                timeout_seconds=args.recovery_timeout_seconds,
            )
            _recover(runtime, args, command, replacement=True)
        else:
            runtime.model = _wrap_ddp(runtime.raw_model, runtime.local_rank)
            runtime.model.train()
            if args.checkpoint_interval > 0:
                _save_checkpoint(runtime, args)
            _write_status(runtime, "running")

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
                if not (
                    args.backend == "nccl2"
                    and _is_comm_error(error)
                ):
                    raise
                _log(
                    runtime.rank,
                    f"recoverable communication failure: {error}",
                )
                command = wait_for_recovery_command(
                    Path(args.recovery_command),
                    timeout_seconds=args.recovery_timeout_seconds,
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
        _write_status(
            runtime,
            "failed",
            error=repr(error),
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--backend",
        choices=("nccl2", "nccl"),
        default="nccl2",
        help="nccl2 enables in-place process-group reconfigure",
    )
    parser.add_argument(
        "--recovery-command",
        type=str,
        default="",
        help="JSON command published by DLRover or the local demo supervisor",
    )
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--dist-timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--recovery-timeout-seconds", type=float, default=180.0
    )
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-at-step", type=int, default=-1)
    parser.add_argument(
        "--slow-rank",
        type=int,
        default=-1,
        help="global rank to slow; -1 disables injection",
    )
    parser.add_argument(
        "--slow-mode",
        choices=("cuda-sleep", "cpu-sleep"),
        default="cuda-sleep",
    )
    parser.add_argument("--slow-delay-ms", type=float, default=25.0)
    parser.add_argument("--slow-extra-cycles", type=int, default=10_000_000)
    parser.add_argument("--slow-start-step", type=int, default=60)
    parser.add_argument("--slow-end-step", type=int, default=900)
    return parser


def _validate_parser(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")
    if args.checkpoint_interval < 0:
        raise SystemExit("--checkpoint-interval must be nonnegative")
    if args.fail_rank < -1 or args.fail_rank >= 4:
        raise SystemExit("--fail-rank must be -1, 0, 1, 2, or 3")
    if (args.fail_rank == -1) != (args.fail_at_step == -1):
        raise SystemExit(
            "--fail-rank and --fail-at-step must be set together"
        )
    if args.fail_at_step < -1:
        raise SystemExit("--fail-at-step must be -1 or nonnegative")
    if args.fail_rank >= 0 and args.fail_at_step >= args.steps:
        raise SystemExit("--fail-at-step must be less than --steps")
    if args.fail_rank >= 0 and args.checkpoint_interval <= 0:
        raise SystemExit(
            "failure injection requires a positive checkpoint interval"
        )
    if args.slow_rank < -1 or args.slow_rank >= 4:
        raise SystemExit("--slow-rank must be -1, 0, 1, 2, or 3")
    if args.slow_delay_ms < 0 or args.slow_extra_cycles < 1:
        raise SystemExit("slowdown values must be nonnegative/positive")
    if args.slow_start_step < 0 or args.slow_end_step <= args.slow_start_step:
        raise SystemExit(
            "slow step range must satisfy 0 <= start < end"
        )
    if args.replacement and not args.recovery_command:
        raise SystemExit(
            "--replacement requires --recovery-command"
        )
    if args.replacement and args.backend != "nccl2":
        raise SystemExit("--replacement requires --backend nccl2")


def main() -> None:
    args = build_parser().parse_args()
    _validate_parser(args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not args.recovery_command:
        args.recovery_command = str(
            work_dir / "recovery" / "command.json"
        )
    train(args)


if __name__ == "__main__":
    main()
