from atorch.common.log_utils import default_logger as logger
from atorch.rl.data.data_utils import RLTrainingDataset


class ReplayBuffer:
    def __init__(self, config, element_keys=None):
        self.config = config
        self.element_keys = element_keys
        self.data = {}
        self.num = 0

    # Reset buffer
    def reset(self):
        for k in self.data.keys():
            self.data[k] = []
        self.num = 0

    def add_samples(self, samples):
        assert isinstance(samples, list)
        for sample in samples:
            self.add_sample(sample)

    # Add a sample or update a sample with index.
    def add_sample(self, sample, index=None):
        sample_keys = [k for k in sample.keys()]
        if self.element_keys is not None:
            assert set(sample_keys).issubset(set(self.element_key)), "replay buffer doesn't contains samples key"
        if index is not None:
            sample_exist = True
            for k in sample_keys:
                if len(self.data.get(k, [])) <= index:
                    logger.warning("failed to update a sample with index {}".format(index))
                    sample_exist = False
                    break
            if sample_exist:
                for k in sample_keys:
                    self.data[k][index] = sample.get(k)
        else:
            for k in sample_keys:
                new_sample = sample.get(k)
                if k not in self.data.keys():
                    self.data[k] = [new_sample]
                else:
                    self.data[k].append(new_sample)
            self.num += 1

    # Sync buffer in process_group using allgather.
    def sync(self, process_group=None):
        pass

    # Create a dataset
    def create_dataset(self):
        return RLTrainingDataset(self)
