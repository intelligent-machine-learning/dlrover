import functools

import numpy as np
import torch
from modeling import ModelType, get_vocab_size
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, size, data_size=(16,), input_size=(16,), output_size=(4,)):
        self.size = size
        self.data_size = data_size
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input": np.ones(self.input_size, dtype=np.float32) * idx,
            "label": np.ones(self.output_size, dtype=np.float32),
        }


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
def collate_sentences_lm(samples, input_name, label_name):
    if len(samples) == 0:
        return {}

    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)

    batch = {
        input_name: src_tokens,
        label_name: tgt_tokens,
    }
    return batch


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
class BenchmarkLMDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.sizes = [self.max_source_positions] * self.total_samples

    def __getitem__(self, index):
        length = self.sizes[index]
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples


def get_dataset(model_type, seq_length=128, input_size=16, output_size=8, datasize=1000):
    if model_type == ModelType.TOY:
        return ToyDataset(size=datasize, input_size=input_size, output_size=output_size)
    if model_type == ModelType.GPT2 or model_type == ModelType.LLAMA:
        vocab_size = get_vocab_size(model_type)
        return BenchmarkLMDataset(vocab_size=vocab_size, max_source_positions=seq_length, total_samples=datasize)


def get_dataloader_args(model_type, batch_size=16):
    dataloader_args = {"batch_size": batch_size, "drop_last": True, "shuffle": True, "num_workers": 2}
    if model_type == ModelType.GPT2 or model_type == ModelType.LLAMA:
        input_name = "input_ids"
        label_name = "labels"
        dataloader_args["collate_fn"] = functools.partial(
            collate_sentences_lm, input_name=input_name, label_name=label_name
        )
    return dataloader_args
