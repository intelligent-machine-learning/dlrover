# This registry registers the communicators needed for building a tensor parallel model
# _INTRA_GROUP_SHARDING_COMMUNICATORS (dict): dict resgistering communicators that supports
# intra-group resharding
from functools import wraps

from atorch.distributed.distributed import parallel_group, parallel_group_and_ranks

from .mappings import (
    collect_and_reshard,
    split_first_dim_with_reshuffle_check,
    split_last_dim_gather_first_dim,
    split_last_dim_with_reshuffle_check,
)
from .utils import _compute_tensor_size

_INTRA_GROUP_SHARDING_COMMUNICATORS = dict(
    {
        ((0,), (-1,)): split_last_dim_gather_first_dim,
        ((), (0,)): split_first_dim_with_reshuffle_check,
        ((), (-1,)): split_last_dim_with_reshuffle_check,
    }
)


def get_communicator(prev_shard, cur_shard):
    """Given a pair of sharding specs (prev_shard, cur_shard), this function find/construct the correct
    function that shards the an input of spec prev_shard into an output of spec cur_shard

    Example:
        >>> prev_shard = MeshShardingSpec(dims=(0), group='model')
        >>> cur_shard = MeshShardingSpec(dims=(-1), group='model')
        >>> resharding_operator = get_communicator(prev_shard, cur_shard)
        >>> resharding_operator
        >>> split_last_dim_gather_first_dim

    There can be cases where the output of a node is not a tensor, but a list/tuple/dict of tensors
    In this case, prev_shard and cur_shard are of the same type as the output, i.e list/tuple/dict of sharding_specs
    get_communicator recursively unwraps the iterable and assigns correct sharding function for each element

    Returns:
        a Callable that shards the input_ of spec: prev_shard into an output of spec: cur_shard
        When no resharding is needed, an identity function is inserted
    """

    def identity_resharding(input_):
        """This resharding operator does nothing"""
        return input_

    if isinstance(prev_shard, list):

        def _communicator(input_):
            resharded_input_ = list(
                get_communicator(p_shard, c_shard)(inp) for p_shard, c_shard, inp in zip(prev_shard, cur_shard, input_)
            )
            return resharded_input_

        return _communicator
    elif isinstance(prev_shard, tuple):

        def _communicator(input_):
            resharded_input_ = tuple(
                get_communicator(p_shard, c_shard)(inp) for p_shard, c_shard, inp in zip(prev_shard, cur_shard, input_)
            )
            return resharded_input_

        return _communicator
    elif isinstance(prev_shard, dict):

        def _communicator(input_):
            resharded_input_ = dict(
                (k, get_communicator(prev_shard[k], cur_shard[k])(inp)) for k, inp in input_.items()
            )
            return resharded_input_

        return _communicator
    elif isinstance(prev_shard, slice):
        # a slice object cannot be sharded, return None
        return identity_resharding
    else:
        # the output of previous node is not a tensor, do not shard it
        if prev_shard is None or cur_shard is None:
            return identity_resharding

        prev_group = prev_shard.group
        cur_group = cur_shard.group
        prev_ranks = prev_shard.ranks
        cur_ranks = cur_shard.ranks
        prev_group = "tensor" if prev_group is None else prev_group
        cur_group = "tensor" if cur_group is None else cur_group

        if prev_group is not None and isinstance(prev_group, str):
            _, prev_ranks = parallel_group_and_ranks(prev_group)
            prev_group = parallel_group(prev_group)
            prev_shard.ranks = prev_ranks
            prev_shard.group = prev_group

        if cur_group is not None and isinstance(cur_group, str):
            _, cur_ranks = parallel_group_and_ranks(cur_group)
            cur_group = parallel_group(cur_group)
            cur_shard.ranks = cur_ranks
            cur_shard.group = cur_group

        if prev_shard == cur_shard:
            return identity_resharding

        # FIXME support inter process group communication
        if set(prev_ranks) != set(cur_ranks):
            raise ValueError("previous node and current node lays on different group")

        def wrap_collect_and_reshard(prev_shard, cur_shard):
            @wraps(collect_and_reshard)
            def _intra_group_collect_and_reshard(input_):
                return collect_and_reshard(input_, prev_shard=prev_shard, cur_shard=cur_shard)

            return _intra_group_collect_and_reshard

        # some operators needs the information of ranks
        # FIXME a hack here, use inspect to make sure operators gets the correct inputs
        def wrap_intra_group_sharding_communicator(func, sharding_spec):
            import inspect

            input_names = inspect.signature(func).parameters.keys()

            @wraps(func)
            def resharding_operator_with_group(input_):
                if "ranks" in input_names:
                    return func(input_, group=sharding_spec.group, ranks=sharding_spec.ranks)
                else:
                    return func(input_, group=sharding_spec.group)

            return resharding_operator_with_group

        # FIXME Stop using these intra_group operators, use collect_and_reshard for all specs?
        if prev_ranks == cur_ranks:
            # No reshuffle, not inter-group, use intra-group operators

            func = _INTRA_GROUP_SHARDING_COMMUNICATORS.get(
                (prev_shard.dims, cur_shard.dims), wrap_collect_and_reshard(prev_shard, cur_shard)
            )
            if (prev_shard.dims, cur_shard.dims) in _INTRA_GROUP_SHARDING_COMMUNICATORS:
                # wrap func to specify the group
                wrapped_intra_group_sharding_communicator = wrap_intra_group_sharding_communicator(func, prev_shard)
                return wrapped_intra_group_sharding_communicator
        else:
            # reshuffle/inter-group, use the general operator
            func = wrap_collect_and_reshard(prev_shard, cur_shard)
        return func


def register_intra_group_communicator(shard_dims_tuple, communicator):
    """Registers a intr-group communicator.

    Args:
        shard_dims_tuple: element 0 is a tuple describing the sharded dimension of the input_ to the
            communicator. Element 1 is a tuple describing the sharded dimension of the output of this
            communicator.
        communicator (Callable): the communicator.
    """
    _INTRA_GROUP_SHARDING_COMMUNICATORS[shard_dims_tuple] = communicator


# For tensor_shape we assume the unsharded tensor size
def get_all_to_all_cost(device_topo, tensor_shape):
    return get_all_gather_cost(device_topo, tensor_shape)


def get_split_cost(device_topo, tensor_shape):
    return 0


def get_all_gather_cost(device_topo, tensor_shape):
    # We assume that tensor_shape is the unsharded tensor
    message_size = _compute_tensor_size(tensor_shape)
    sharded_message_size = message_size / device_topo.num_devices()
    device_ranks = device_topo.get_device_ranks()
    # calculate the maximum communication time between any two devices
    max_communication_time = max(
        sharded_message_size / device_topo.get_effective_bandwidth(device_id_1, device_id_2)
        for device_id_1 in device_ranks
        for device_id_2 in device_ranks
    )

    return max_communication_time


def get_all_reduce_cost(device_topo, tensor_shape):
    return get_all_gather_cost(device_topo, tensor_shape)


_COMMUNICATOR_COSTS = dict(
    {"all_reduce": get_all_reduce_cost, "all_to_all": get_all_to_all_cost, "split": get_split_cost}
)
