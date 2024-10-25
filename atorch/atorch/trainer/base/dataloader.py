from torch.utils.data import DataLoader


class AtorchDataloader(DataLoader):
    def __init__(self, dataset, sampler, **kwargs):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        pass

    def __len__(self):
        pass
