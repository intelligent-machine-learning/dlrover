from contextlib import contextmanager

import torch

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import _DistributedContext as dc


class _Randomizer:
    """
    Torch random number generator (both cuda and cpu) state tracker.
    Init with seeded state, and track the state under fork contextmanager.
    """

    def __init__(self, seed):
        self.seed = seed
        # cuda rng
        ori_cuda_rng = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(self.seed)
        self.cuda_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(ori_cuda_rng)
        # cpu rng
        ori_cpu_rng = torch.get_rng_state()
        torch.manual_seed(self.seed)
        self.cpu_rng = torch.get_rng_state()
        torch.set_rng_state(ori_cpu_rng)

    @contextmanager
    def fork(self):
        ori_cuda_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.cuda_rng)
        ori_cpu = torch.get_rng_state()
        torch.set_rng_state(self.cpu_rng)
        try:
            yield
        finally:
            self.cuda_rng = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(ori_cuda_rng)
            self.cpu_rng = torch.get_rng_state()
            torch.set_rng_state(ori_cpu)


class MultiDimParallelRandomizer:
    """
    Multiple dimension parallel randomizers manager that handles the same/different seeded states.
    """

    def __init__(self, base_seed):
        if not dc.INITIALIZED:
            logger.warning("_DistributedContext not initialized.")
            return
        self.base_seed = base_seed
        # parallel group info
        self.parallel_group_names = list(dc.PARALLEL_GROUP.keys())
        self.parallel_group_sizes = [dc.PARALLEL_GROUP_SIZE[n] for n in self.parallel_group_names]
        self.parallel_ranks = [dc.PARALLEL_RANK[n] for n in self.parallel_group_names]
        # seed offset multiply factor
        self.m_factor = [1]
        for size in self.parallel_group_sizes:
            self.m_factor.append(size * self.m_factor[-1])

        self._randomizers = dict()

    def get_randomizer(self, *same_groups):
        """
        Get the randomizer for same_groups. Every same_groups configuration initializes its randomizer
        in the first call and stored in _randomizers.
        Use same_tuple as _randomizers' key, default all False and turned True if group name in same_groups.

        Arguments::
        - same_groups: any number of parallel group names. Each assigned group use the same seed to start tracked rng.

        Example::
           >>> # parallel_group_names = ['tenosr', 'data', 'pipeline']
           >>> # initializing weight (assume tp parts are initialized in whole and further splitted, thus in same)
           >>> with get_randomizer("tensor", "data"):
           >>>     get_model()
           >>> # dropout of replica inputs, tp needs the same dropout pattern
           >>> with get_randomizer("tensor"):
           >>>     drop(m)
           >>> # dropout of parallel input, needing all the different seed
           >>> with get_randomizer():
           >>>     drop(m)
        """
        assert all(
            name in self.parallel_group_names for name in same_groups
        ), f"same_groups {same_groups} has elements not in parallel_group_names {self.parallel_group_names}"
        same_tuple = tuple(True if name in same_groups else False for name in self.parallel_group_names)
        if same_tuple in self._randomizers:
            return self._randomizers[same_tuple]

        # init the randomizer for this same_tuple
        same_code = sum(2**i * int(same) for i, same in enumerate(same_tuple))
        offset = same_code * self.m_factor[-1]
        for i, same in enumerate(same_tuple):
            offset += 0 if same else self.parallel_ranks[i] * self.m_factor[i]
        seed = self.base_seed + offset
        self._randomizers[same_tuple] = _Randomizer(seed)
        return self._randomizers[same_tuple]

    def get_states(self):
        states = {
            same_tuple: {
                "cuda_rng": _randomizer.cuda_rng,
                "cpu_rng": _randomizer.cpu_rng,
            }
            for same_tuple, _randomizer in self._randomizers.items()
        }
        return states

    def set_states(self, states):
        assert set(self._randomizers.keys()) == set(states.keys()), (
            f"Keys mismatch, self._randomizers: {set(self._randomizers.keys())}, " f"states: {set(states.keys())}."
        )
        for name in states:
            self._randomizers[name].cuda_rng = states[name]["cuda_rng"]
            self._randomizers[name].cpu_rng = states[name]["cpu_rng"]


# MultiDimParallelRandomizer Singleton Instance
_MDPRInstance = None


def get_MDPRInstance():
    global _MDPRInstance
    assert _MDPRInstance is not None, "Multiple dimension parallel randomizer not initialized."
    return _MDPRInstance


def init_randomizer(base_seed=1234):
    global _MDPRInstance
    assert _MDPRInstance is None, "Repeatedly initializing multiple dimension parallel randomizer."
    _MDPRInstance = MultiDimParallelRandomizer(base_seed)


def get_randomizer(*same_groups):
    """
    See `MultiDimParallelRandomizer.get_randomizer`
    """
    global _MDPRInstance
    assert _MDPRInstance is not None, "Multiple dimension parallel randomizer not initialized."
    return _MDPRInstance.get_randomizer(*same_groups)
