class AtorchDataset:
    def from_config(cls, distributed_type, *args, **kwargs) -> "AtorchDataset":
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class OdpsDataset(AtorchDataset):
    def __init__(self):
        pass


class PcacheDataset(AtorchDataset):
    def __init__(self):
        pass
