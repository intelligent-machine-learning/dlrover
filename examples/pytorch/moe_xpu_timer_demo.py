#!/usr/bin/env python3
"""Four-GPU MoE job for testing DLRover performance diagnosis.

DLRover owns the xpu_timer lifecycle. This script only emits structured
training metrics that DLRover can parse.

The default model is intentionally small. Use ``--slow-rank`` to inject a
repeatable slowdown into one rank while the other ranks execute the same
workload. ``cuda-sleep`` inserts the same CUDA sleep kernel on every rank and
only changes its duration on the selected rank. ``cpu-sleep`` stalls one host
before the forward pass.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


BATCH_SIZE = 32
SEQ_LEN = 64
VOCAB_SIZE = 1_000
EMBED_DIM = 128
NUM_EXPERTS = 4
TOP_K = 1


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


def setup_ddp() -> Tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError(
            "This demo is intended for exactly four ranks; "
            f"torchrun reported WORLD_SIZE={world_size}."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


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
                output[mask] += expert(flat_x[mask]) * expert_weight[mask]

        top1 = torch.argmax(probs, dim=-1)
        density = (
            torch.bincount(top1, minlength=self.num_experts).float()
            / top1.numel()
        )
        aux_loss = (
            (self.num_experts * density)
            * (self.num_experts * probs.mean(dim=0))
        ).mean()
        return (
            output.reshape(batch_size, seq_len, embed_dim),
            aux_loss,
        )


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
    def __init__(self, sequence_length: int, length: int = 10_000) -> None:
        self.sequence_length = sequence_length
        self.length = length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        del index
        tokens = torch.randint(0, VOCAB_SIZE, (self.sequence_length,))
        return tokens, tokens

    def __len__(self) -> int:
        return self.length


def inject_slowdown(plan: SlowPlan, rank: int, step: int, device: int) -> None:
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
    """Estimate top-1 MoE forward and backward FLOPs per token."""

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
    """Print the format consumed by PerformanceDiagnostician."""

    interval_ms = interval_seconds * 1_000.0
    elapsed_ms_per_iteration = interval_ms / interval_steps
    tokens_per_gpu = interval_steps * BATCH_SIZE * SEQ_LEN
    tokens_per_second_per_gpu = tokens_per_gpu / max(1e-9, interval_seconds)
    tflops_per_gpu = (
        tokens_per_second_per_gpu * estimated_training_flops_per_token() / 1e12
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


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    plan = SlowPlan(
        rank=args.slow_rank,
        mode=args.slow_mode,
        delay_ms=args.slow_delay_ms,
        extra_cycles=args.slow_extra_cycles,
        start_step=args.slow_start_step,
        end_step=args.slow_end_step,
    )

    torch.manual_seed(1234)
    model = DDP(
        ToyMoETransformer().to(device),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataset = LazyRandomDataset(SEQ_LEN)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    if rank == 0:
        print(
            "MoE DLRover diagnosis demo: "
            f"world_size={world_size}, steps={args.steps}, "
            f"slow_rank={args.slow_rank}, mode={args.slow_mode}, "
            f"active_steps=[{args.slow_start_step},{args.slow_end_step})",
            flush=True,
        )

    model.train()
    try:
        global_step = 0
        epoch = 0
        interval_steps = 0
        interval_start = time.perf_counter()
        last_loss = 0.0
        while global_step < args.steps:
            sampler.set_epoch(epoch)
            for inputs, targets in loader:
                if global_step >= args.steps:
                    break

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                inject_slowdown(plan, rank, global_step, local_rank)

                optimizer.zero_grad(set_to_none=True)
                logits, aux_loss = model(inputs)
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                )
                (loss + 0.01 * aux_loss).backward()
                optimizer.step()

                global_step += 1
                interval_steps += 1
                last_loss = loss.item()
                should_log = (
                    interval_steps >= args.log_every
                    or global_step >= args.steps
                )
                if rank == 0 and should_log:
                    torch.cuda.synchronize(local_rank)
                    interval_seconds = time.perf_counter() - interval_start
                    print_performance_metrics(
                        global_step=global_step,
                        total_steps=args.steps,
                        loss_value=last_loss,
                        interval_steps=interval_steps,
                        interval_seconds=interval_seconds,
                    )
                    interval_start = time.perf_counter()
                    interval_steps = 0
            epoch += 1
    finally:
        cleanup_ddp()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help=(
            "emit a DLRover-compatible performance record after this many "
            "steps"
        ),
    )
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
    parser.add_argument(
        "--slow-delay-ms",
        type=float,
        default=25.0,
        help="host delay for --slow-mode cpu-sleep",
    )
    parser.add_argument(
        "--slow-extra-cycles",
        type=int,
        default=10_000_000,
        help="CUDA sleep cycles on the selected rank",
    )
    parser.add_argument("--slow-start-step", type=int, default=60)
    parser.add_argument("--slow-end-step", type=int, default=900)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")
    if args.slow_rank < -1 or args.slow_rank >= 4:
        raise SystemExit("--slow-rank must be -1, 0, 1, 2, or 3")
    if args.slow_delay_ms < 0 or args.slow_extra_cycles < 1:
        raise SystemExit("slowdown values must be nonnegative/positive")
    if args.slow_start_step < 0 or args.slow_end_step <= args.slow_start_step:
        raise SystemExit("slow step range must satisfy 0 <= start < end")
    train(args)


if __name__ == "__main__":
    main()
