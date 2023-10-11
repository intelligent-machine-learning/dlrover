import torch

try:
    from torch.distributed._shard.sharding_spec import ShardingSpec
except ImportError:
    ShardingSpec = object
from atorch.distributed.distributed import parallel_group_and_ranks


class MeshShardingSpec(ShardingSpec):
    def __init__(self, dims, group=None, ranks=None):
        self.dims = dims
        self.group = group
        self.ranks = ranks

    # Since ranks and group has a one-one correspondence, throw away trouble some ranks
    def __eq__(self, other):
        return self.dims == other.dims and self.group == other.group

    def __ne__(self, other):
        return self.dims != other.dims or self.group != other.group

    def __hash__(self):
        return hash((self.dims, self.group))

    def shard(self, tensor, src_rank, process_group=None):
        pass

    def build_metadata(self, tensor_sizes, tensor_properties):
        pass

    def __str__(self):
        output = f"shard dim: {self.dims}, shard ranks: {self.ranks}, shard group: {self.group}"
        return output

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["group"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # see atorch.modules.distributed_modules.mapping_registry get_communicator method
        # Sharding specs with group being str ranks being None will be automatically completed,
        # to allow it correctly find the ranks, need this to be an str identifier instead of the real process group.
        self.__dict__["group"] = "tensor"


def _extract_sharding_spec(result, spec_name="mesh"):
    """
    Extract a sharding spec for `result`.

    Args:
        result: an element of the output of a node.
        spec_name: the sharding spec to use, currently only support MeshShardingSpec.

    Returns:
        The sharding spec of the result, a MeshShardingSpec if result is a Tensor,
            None otherwise.
    """
    if isinstance(result, torch.Tensor):
        shape = result.shape
        # assume not sharded
        if len(shape) == 0 or len(shape) == 1:
            return None
        else:
            if spec_name == "mesh":
                return MeshShardingSpec(dims=tuple(), group="tensor", ranks=None)
            else:
                raise ValueError(f"spec: {spec_name} is currently not supported")
    else:
        return None


def shard_eq(prev_shard, cur_shard):
    """Test if prev_shard and cur_shard are equal (so no communicator need to be inserted)
    sharding_specs could be wrapped recursively in tuple/list/dict
    to test the equality of the two, unwrap the container and assign correctly the group

    Args:
        prev_shard: sharding spec possibly wrapped
        cur_shard: sharding spec possibly wrapped

    Returns
        whether two shards are equivalent in the sense no communicator is needed
    """
    if isinstance(prev_shard, (list, tuple)):
        return all(list(shard_eq(p_shard, c_shard) for p_shard, c_shard in zip(prev_shard, cur_shard)))
    elif isinstance(prev_shard, dict):
        return all(list(shard_eq(prev_shard[key], cur_shard[key]) for key in prev_shard.keys()))
    elif isinstance(prev_shard, slice):
        # a slice object cannot be sharded, return equal
        return True
    else:
        if (
            prev_shard is None
            or cur_shard is None
            or not isinstance(prev_shard, MeshShardingSpec)
            or not isinstance(cur_shard, MeshShardingSpec)
        ):
            return True

        prev_group = prev_shard.group
        cur_group = cur_shard.group
        prev_ranks = prev_shard.ranks
        cur_ranks = cur_shard.ranks
        prev_group = "tensor" if prev_group is None else prev_group
        cur_group = "tensor" if cur_group is None else cur_group

        if prev_group is not None and isinstance(prev_group, str):
            prev_group, prev_ranks = parallel_group_and_ranks(prev_group)
            prev_shard.ranks = prev_ranks
            prev_shard.group = prev_group

        if cur_group is not None and isinstance(cur_group, str):
            cur_group, cur_ranks = parallel_group_and_ranks(cur_group)
            cur_shard.ranks = cur_ranks
            cur_shard.group = cur_group
        return prev_shard == cur_shard
