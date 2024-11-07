import os
from datetime import timedelta

import torch
import torch.distributed as dist

from atorch.trainer.utils import DistributedType
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core import mpu


class AtorchAcceleratorState:
    def __init__(self, args):
        self.distributed_type = None
        # Unify torch.distributed.init_process_group()
        device_count = torch.cuda.device_count()
        device = self.process_index % device_count
        if self.local_process_index is not None:
            assert self.local_process_index == device, "expected local-rank to be the same as rank % device-count."
        else:
            self.local_process_index = device
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            # Call the init process
            dist.init_process_group(
                backend=args.ddp_backend,
                world_size=self.num_processes,
                rank=self.process_index,
                timeout=timedelta(seconds=args.ddp_timeout),
            )
        if args.distributed_type == "ddp":
            self.distributed_type = DistributedType.MULTI_GPU
        elif args.distributed_type == "fsdp":
            self.distributed_type = DistributedType.FSDP
            raise ValueError("Not implement FSDP.")
        elif args.distributed_type == "deepspeed":
            self.distributed_type = DistributedType.DEEPSPEED
            raise ValueError("Not implement DeepSpeed.")
        elif args.distributed_type == "megatron":
            self.distributed_type = DistributedType.MEGATRON
        else:
            raise ValueError("Not supported distributed type.")

    @property
    def initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def deveice(self):
        pass

    @property
    def num_processes(self) -> int:
        if dist.is_initialized():
            return dist.get_world_size()
        return int(os.getenv("WORLD_SIZE", "1"))

    @property
    def process_index(self) -> int:
        if dist.is_initialized():
            return dist.get_rank()
        return int(os.getenv("RANK", "0"))

    @property
    def local_process_index(self) -> int:
        return int(os.environ.get("LOCAL_RANK", -1))

    @property
    def is_last_process(self) -> bool:
        return self.process_index == self.num_processes - 1

    @property
    def is_main_process(self) -> bool:
        return self.process_index == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_process_index == 0

    @property
    def dp_group_size(self) -> int:
        if self.distributed_type == DistributedType.MEGATRON:
            return mpu.get_data_parallel_world_size()
        return self.num_processes
