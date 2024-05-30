# Copyright 2024 The DLRover Authors. All rights reserved.
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

from abc import ABCMeta, abstractmethod
from typing import Dict, List

import torch
import torch.distributed as dist

from dlrover.python.common import env_utils
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    DLROVER_CKPT_CONFIG_KEY,
    CheckpointConfig,
    SharedMemoryHandler,
)


class BackupManger(metaclass=ABCMeta):
    def __init__(self, local_rank, local_world_size) -> None:
        self.local_rank = local_rank
        self.local_world_size = local_world_size
        self.node_rank = env_utils.get_node_rank()
        self.rank = dist.get_rank()
        self.node_num = env_utils.get_node_num()
        self.current_device = torch.device("cpu")

    @abstractmethod
    def backup(self, shm_handler: SharedMemoryHandler):
        """
        The nodes in a backup group back up the checkpoint of each other.
        """
        pass

    @abstractmethod
    def gather(self):
        """
        The node gather the checkpoint from the memory of other node in
        a backup group.
        """
        pass


class ShardCkptBackupManager(BackupManger):
    """
    The manager will select a rank of another node to backup the checkpoint
    of the current rank.
    """

    def __init__(
        self, local_rank, local_world_size, backup_group_size=2
    ) -> None:
        super().__init__(local_rank, local_world_size)

        self.backup_ranks = self._get_backup_ranks(backup_group_size)
        self._backup_group = dist.new_group(
            backend="gloo", ranks=self.backup_ranks
        )
        self._rank_shms: Dict[int, SharedMemoryHandler] = {}

    def _get_backup_ranks(self, backup_group_size):
        """
        Get the ranks to backup checkpoint. Assuming each group has 3 nodes
        (group_size=3) and each node has 2 ranks. The backup ranks of local
        rank in each node are:
        local rank 0: {0, 8 ,16}
        local rank 1: {1, 9, 17}

        Arguments:
            node_rank: the rank of node in the job.
            local_rank: the local rank in a node.
            local_world_size: the number of local ranks in a node.
            group_size: the number of nodes in each backup group.

        Returns:
            A list of ranks.
        """
        backup_ranks = []

        group_index = self.node_rank // backup_group_size
        for i in range(backup_group_size):
            node_rank = group_index * backup_group_size + i
            rank = node_rank * self.local_world_size + self.local_rank
            backup_ranks.append(rank)
        return backup_ranks

    def backup(self, shm_handler: SharedMemoryHandler):
        """
        The rank of node in a backup group send its checkpoint shard
        in the shared memory to other nodes and get the checkpoint shards
        of other nodes by allgather.
        """
        assert shm_handler.shared_memory is not None
        buffer = shm_handler.shared_memory.buf
        meta_data = shm_handler.metadata.get()
        shm_tensors, ckpt_metas = self._gather_peer_ckpt(buffer, meta_data)
        self._rank_shms[self.rank] = shm_handler
        self._write_peer_ckpt_to_shm(shm_tensors, ckpt_metas)

    def _gather_peer_ckpt(self, buffer, meta_data):
        byte_tensor = torch.ByteTensor(buffer)
        local_size = torch.LongTensor([byte_tensor.numel()]).to(
            self.current_device
        )

        group_size = dist.get_world_size(group=self._backup_group)

        shm_sizes_tensor = torch.zeros(
            group_size, dtype=torch.long, device=self.current_device
        )
        shm_size_list = [
            shm_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
        ]
        # Allgather tensor sizes
        dist.all_gather(shm_size_list, local_size, group=self._backup_group)

        max_tensor_size = int(max(shm_size_list))
        # Resize tensor to max size across all ranks.
        byte_tensor.resize_(max_tensor_size)

        output_tensors = [
            torch.empty(
                max_tensor_size, dtype=torch.uint8, device=self.current_device
            )
            for _ in shm_size_list
        ]
        dist.all_gather(output_tensors, byte_tensor, group=self._backup_group)

        output_meta_objs = [None for _ in range(group_size)]
        dist.all_gather_object(output_meta_objs, meta_data)
        return output_tensors, output_meta_objs

    def _write_peer_ckpt_to_shm(self, shm_tensors, ckpt_metas):
        for shm_tensor, meta in zip(shm_tensors, ckpt_metas):
            # Skip writing the self checkpoint into the shared memory.
            config: CheckpointConfig = meta[DLROVER_CKPT_CONFIG_KEY]
            if config.rank == self.rank:
                continue
            if config.rank not in self._rank_shms:
                shm_hanlder = SharedMemoryHandler(local_rank=config.rank)
                shm_hanlder.init_shared_memory(
                    create=True, size=shm_tensor.numel()
                )
                self._rank_shms[config.rank] = shm_hanlder
            shm_hanlder = self._rank_shms[config.rank]
            local_shm_tensor = torch.frombuffer(
                buffer=shm_hanlder.shared_memory.buf,
                dtype=torch.uint8,
            )
            local_shm_tensor.copy_(shm_tensor)
            shm_hanlder.metadata.set(meta)

    def gather(self):
        """
        The method gathers the checkpoint shard from the memory of the peer
        node in a backup group. Assuming each backup group has two nodes,
        the each rank of each node has two checkpoint shards. For example,
        assuming each node only has one rank, the checkpoint shards of 2 nodes
        is like {0: shard_0, 1: shard_1}. The method will perform
        len(ckpt_shards) rounds allgather for each shard. In the 1st round,
        the ranks in 2 nodes allgather shard_0 and rank-0 will select the
        valid shard_0 from all ranks. In the 2nd round, the rank-1 will select
        the valid shard_1 from all ranks.

        Arguments:
            ckpt_shards (dict): the key is the rank of checkpoint shard and the
                value is the handle fo the shared memory to store the
                checkpoint shard.
            ckpt_metas (dict): the key is the rank of checkpoint shard and the
                value is the meta dict of PyTorch checkpiont state dict.

        Returns:
            ByteTensor of the checkpoint shard.
            A dict of checkpoint shard meta data.
        """

        shm_handlers = {}
        for rank in self.backup_ranks:
            if rank == self.rank:
                shm_handler = SharedMemoryHandler(local_rank=self.local_rank)
            else:
                shm_handler = SharedMemoryHandler(local_rank=rank)
            shm_handler.init_shared_memory()
            shm_handlers[rank] = shm_handler
        shm_tensor, meta = self._gather_owner_checkpoint(shm_handlers)
        return shm_tensor, meta

    def _gather_owner_checkpoint(
        self, shm_handlers: List[SharedMemoryHandler]
    ):
        ckpt_shm_tensor = None
        ckpt_meta = {}
        for i, rank in enumerate(self.backup_ranks):
            shm_handler = shm_handlers[i]
            if shm_handler.shared_memory:
                assert shm_handler.shared_memory is not None
                buffer = shm_handler.shared_memory.buf
                meta_data = shm_handlers[rank].metadata.get()
            else:
                buffer = memoryview(b"")
                meta_data = {}
            shm_tensors, ckpt_metas = self._gather_peer_ckpt(buffer, meta_data)
            if rank != self.rank:
                continue
            for shm_tenor, meta in zip(shm_tensors, ckpt_metas):
                if meta:
                    ckpt_shm_tensor = shm_tenor
                    ckpt_meta = meta
                    break
        return ckpt_shm_tensor, ckpt_meta


class FullCkptBackupManager(BackupManger):
    """
    The node does not need to backup checkpoint if each rank has
    the full checkpoint. The manager can select one rank which
    has the full checkpoint to broadcast the checkpiont in the
    shared memory to other ranks.
    """

    def __init__(self, local_rank, local_world_size) -> None:
        super().__init__(local_rank, local_world_size)
        self.backup_ranks = self._get_backup_ranks()
        self._backup_group = dist.new_group(
            backend="gloo", ranks=self.backup_ranks
        )
        self._rank_shms: Dict[int, SharedMemoryHandler] = {}

    def _get_backup_ranks(self):
        backup_ranks = []
        for node_rank in range(self.node_num):
            rank = node_rank * self.local_world_size
            backup_ranks.append(rank)
        return backup_ranks

    def backup(self, shm_handler: SharedMemoryHandler):
        """
        Each rank in the node has full model checkpoint. The checkpoints
        in the shared memory of each node are backups of the checkpoint.
        """
        pass

    def gather(self):
        """
        The method gathers the checkpoint shard from the memory of the peer
        node in a backup group. Firstly, the method select a source rank
        in a node whose shared memory has the complete checkpoint. Then
        the rank broadcast its checkpoit to other ranks.
        Arguments:
            ckpt_shards (dict): the key is the rank of checkpoint shard and the
                value is the handle fo the shared memory to store the
                checkpoint shard.
            ckpt_metas (dict): the key is the rank of checkpoint shard and the
                value is the meta dict of PyTorch checkpiont state dict.

        Returns:
            ByteTensor of the checkpoint shard.
            A dict of checkpoint shard meta data.
        """

        shm_handlers = {}
        if self.rank not in self.backup_ranks:
            return

        shm_handler = SharedMemoryHandler(local_rank=self.local_rank)
        shm_handler.init_shared_memory()
        shm_tensor, meta = self._gather_owner_checkpoint(shm_handlers)
        return shm_tensor, meta

    def _gather_owner_checkpoint(self, shm_handler: SharedMemoryHandler):
        flag = torch.tensor([1], dtype=torch.int8)
        if not shm_handler.shared_memory:
            flag = torch.tensor([0], dtype=torch.int8)
        output_tensors = [
            torch.empty(1, dtype=torch.int8) for _ in self.backup_ranks
        ]
        dist.all_gather(output_tensors, flag)
        src_rank = self.backup_ranks[0]
        for i, flag in enumerate(output_tensors):
            if flag == 1:
                src_rank = self.backup_ranks[i]
                break
        if shm_handler.shared_memory:
            byte_tensor = torch.ByteTensor(shm_handler.shared_memory.buf)
        else:
            byte_tensor = torch.ByteTensor([])
        dist.broadcast(byte_tensor, src=src_rank)
        ckpt_meta = shm_handler.metadata.get()
        ckpt_metas = [ckpt_meta]
        dist.broadcast_object_list(ckpt_metas, src=src_rank)
        return byte_tensor, ckpt_metas[0]
