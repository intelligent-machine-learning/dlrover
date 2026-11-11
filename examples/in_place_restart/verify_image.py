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
"""Verify that a container can run the in-place restart example."""

import argparse
import inspect
import os
import secrets
import shutil
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist


MIN_TORCH = (2, 14)
MIN_NCCL = (2, 28, 0)
MIN_SHM_GB = 4.0


def _version_tuple(value: str) -> Tuple[int, int]:
    parts = value.split("+")[0].split(".")
    return int(parts[0]), int(parts[1])


def _check(label: str, condition: bool, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"[{'PASS' if condition else 'FAIL'}] {label}{suffix}")
    if not condition:
        raise RuntimeError(f"{label} check failed{suffix}")


def _check_local_environment(world_size: int) -> None:
    torch_version = _version_tuple(torch.__version__)
    _check(
        "PyTorch >= 2.14",
        torch_version >= MIN_TORCH,
        torch.__version__,
    )
    _check(
        "CUDA is available",
        torch.cuda.is_available(),
        f"build CUDA {torch.version.cuda}",
    )
    _check(
        f"at least {world_size} CUDA devices",
        torch.cuda.device_count() >= world_size,
        f"found {torch.cuda.device_count()}",
    )

    nccl_version = tuple(torch.cuda.nccl.version())
    _check(
        "NCCL >= 2.28 for nccl2 reconfigure",
        nccl_version >= MIN_NCCL,
        ".".join(str(value) for value in nccl_version),
    )
    _check(
        "nccl2 backend is available",
        dist.is_backend_available("nccl2"),
    )

    required_apis = (
        "_supports_reconfigure",
        "_get_reconfigure_handle",
        "_reconfigure",
        "get_backend_impl",
    )
    missing = [name for name in required_apis if not hasattr(dist, name)]
    _check(
        "process-group reconfigure APIs exist",
        not missing,
        f"missing={missing}" if missing else "",
    )
    init_parameters = inspect.signature(
        dist.init_process_group
    ).parameters
    _check(
        "init_process_group accepts enable_reconfigure",
        "enable_reconfigure" in init_parameters,
    )

    shared_memory = os.statvfs("/dev/shm")
    shared_memory_gb = (
        shared_memory.f_bavail * shared_memory.f_frsize
    ) / (1024**3)
    _check(
        f"/dev/shm >= {MIN_SHM_GB:g} GiB",
        shared_memory_gb >= MIN_SHM_GB,
        f"{shared_memory_gb:.2f} GiB",
    )

    dlrover_run = shutil.which("dlrover-run")
    _check("dlrover-run is installed", dlrover_run is not None)
    dlrover_help = subprocess.run(
        [dlrover_run, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    _check(
        "dlrover-run starts successfully",
        dlrover_help.returncode == 0,
        dlrover_help.stderr.strip()[:300],
    )
    try:
        import dlrover  # noqa: F401
    except ImportError as error:
        raise RuntimeError("the dlrover package is not importable") from error
    print("[PASS] dlrover package imports")

    example_dir = Path(__file__).resolve().parent
    required_files = (
        "moe_train.py",
        "recovery_protocol.py",
        "run_in_place_demo.py",
    )
    missing_files = [
        name
        for name in required_files
        if not (example_dir / name).is_file()
    ]
    _check(
        "example worker files exist",
        not missing_files,
        f"missing={missing_files}" if missing_files else "",
    )

    print("")
    for rank in range(world_size):
        print(
            f"GPU {rank}: {torch.cuda.get_device_name(rank)}"
        )


def _distributed_worker(
    rank: int,
    store_path: str,
    world_size: int,
) -> None:
    torch.cuda.set_device(rank)
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="nccl2",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
        enable_reconfigure=True,
    )

    try:
        default_group = dist.distributed_c10d._get_default_group()
        _check(
            f"rank {rank}: nccl2 supports reconfigure",
            bool(dist._supports_reconfigure(default_group)),
        )

        prefix = dist.PrefixStore("verify-image", store)
        prefix.set(
            f"handle/{rank}",
            dist._get_reconfigure_handle(default_group),
        )
        handle_keys = [
            f"handle/{handle_rank}"
            for handle_rank in range(world_size)
        ]
        prefix.wait(handle_keys, timedelta(seconds=60))
        handles = [
            prefix.get(key).decode("utf-8") for key in handle_keys
        ]

        uuid_key = "verify-image/uuid"
        if rank == 0:
            store.set(uuid_key, str(secrets.randbits(62) + 1))
        store.wait([uuid_key], timedelta(seconds=60))
        reconfigure_uuid = int(store.get(uuid_key).decode("utf-8"))

        work = dist._reconfigure(
            uuid=reconfigure_uuid,
            handles=handles,
            group=default_group,
            timeout=timedelta(seconds=60),
        )
        work.wait()

        tensor = torch.ones(1, device=torch.device("cuda", rank))
        dist.all_reduce(tensor)
        _check(
            f"rank {rank}: reconfigured all-reduce",
            tensor.item() == world_size,
            f"sum={tensor.item()}",
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _check_distributed(world_size: int) -> None:
    with tempfile.TemporaryDirectory(
        prefix="dlrover-image-check-"
    ) as directory:
        store_path = str(Path(directory) / "store")
        try:
            torch.multiprocessing.spawn(
                _distributed_worker,
                args=(store_path, world_size),
                nprocs=world_size,
                join=True,
            )
        except Exception as error:
            raise RuntimeError(
                "four-rank nccl2 reconfigure test failed"
            ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--skip-distributed",
        action="store_true",
        help="only check packages, APIs, GPUs, and container limits",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size != 4:
        raise SystemExit("this prototype requires --world-size=4")

    print("Container preflight checks")
    _check_local_environment(args.world_size)
    if not args.skip_distributed:
        print("")
        print("Four-rank nccl2 reconfigure check")
        _check_distributed(args.world_size)
    print("")
    print("IMAGE PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
