import copy
import os
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

import atorch
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import create_parallel_group
from atorch.modules.distributed_modules.cross_entropy import vocab_parallel_cross_entropy
from atorch.modules.distributed_modules.layers import ParallelEmbedding, VocabParallelEmbedding


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)

    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")
    parallel_config = ([("model", world_size)], None)

    create_parallel_group(parallel_config)


class VocabEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, config):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        # And initialize.
        init.xavier_normal_(self.weight)

    def forward(self, input_):
        # Get the embeddings.
        output = F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return output


def _run_megatron_vocab_parallel_embedding(rank, world_size):

    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    class Config:
        vocab_size = 1024
        embedding_dim = 2048
        hidden_size = 1024

    torch.manual_seed(0)
    vocab_embedding = VocabEmbedding(Config).to(device)
    vocab_embedding_copy = copy.deepcopy(vocab_embedding)

    parallel_vocab_embedding = VocabParallelEmbedding(
        orig_module=vocab_embedding, process_group=pg, ranks=ranks, defer_init=False
    ).to(device)
    if rank == 0:
        r = vocab_embedding_copy.weight[0:512, :] - parallel_vocab_embedding.weight[:, :]
        assert torch.norm(r, p=-1) == 0
    elif rank == 1:
        r = vocab_embedding_copy.weight[512:1024, :] - parallel_vocab_embedding.weight[:, :]
        assert torch.norm(r, p=-1) == 0
    input_ids = torch.tensor([0, 513], dtype=torch.long).to(device)

    embed1 = vocab_embedding_copy(input_ids)
    embed2 = parallel_vocab_embedding(input_ids)
    assert torch.norm(embed1 - embed2, p=-1) == 0
    atorch.reset_distributed()


def _run_megatron_vocab_parallel_cross_entropy(rank, world_size):

    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    class Config:
        vocab_size = 1024
        embedding_dim = 2048
        hidden_size = 1024

    torch.manual_seed(0)
    vocab_embedding = VocabEmbedding(Config).to(device)
    vocab_embedding_copy = copy.deepcopy(vocab_embedding)

    parallel_vocab_embedding = VocabParallelEmbedding(
        orig_module=vocab_embedding, process_group=pg, ranks=ranks, defer_init=False
    ).to(device)

    batch_size = 4
    seq_len = 512
    torch.manual_seed(1)
    ref_last_hidden_states = torch.randn((batch_size, seq_len, Config.hidden_size), device=device, requires_grad=True)
    tp_last_hidden_states = copy.deepcopy(ref_last_hidden_states)
    ref_logits = F.linear(ref_last_hidden_states, vocab_embedding_copy.weight)
    tp_logits = F.linear(tp_last_hidden_states, parallel_vocab_embedding.weight)

    torch.manual_seed(2)
    labels = torch.randint(
        low=0,
        high=Config.vocab_size,
        size=(
            batch_size,
            seq_len,
        ),
        device=device,
    )
    loss_mask = (
        torch.randn(
            (
                batch_size,
                seq_len,
            ),
            device=device,
        )
        > 0
    )

    # ref loss
    ref_losses = torch.nn.CrossEntropyLoss(reduction="none")(ref_logits.view(-1, ref_logits.size(-1)), labels.view(-1))
    ref_loss = torch.sum(ref_losses * loss_mask.view(-1))
    if loss_mask.sum().item() > 0:
        ref_loss = ref_loss / loss_mask.sum()
    ref_loss.backward()

    # tp loss
    tp_losses = vocab_parallel_cross_entropy(tp_logits, labels)
    loss_mask = loss_mask.view(-1)
    tp_loss = torch.sum(tp_losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        tp_loss = tp_loss / loss_mask.sum()
    tp_loss.backward()

    # check
    assert torch.all(torch.isclose(ref_loss, tp_loss))
    assert torch.all(
        torch.isclose(
            vocab_embedding_copy.weight.grad.chunk(world_size, dim=0)[rank], parallel_vocab_embedding.weight.grad
        )
    )
    atorch.reset_distributed()


def _run_megatron_parallel_embedding(rank, world_size):

    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    class Config:
        vocab_size = 1024
        embedding_dim = 2048
        hidden_size = 1024

    torch.manual_seed(0)
    vocab_embedding = VocabEmbedding(Config).to(device)
    vocab_embedding_copy = copy.deepcopy(vocab_embedding)

    parallel_vocab_embedding = ParallelEmbedding(
        orig_module=vocab_embedding, process_group=pg, ranks=ranks, defer_init=False
    ).to(device)
    if rank == 0:
        r = vocab_embedding_copy.weight[:, 0:512] - parallel_vocab_embedding.weight[:, :]
        assert torch.norm(r, p=-1) == 0
    elif rank == 1:
        r = vocab_embedding_copy.weight[:, 512:1024] - parallel_vocab_embedding.weight[:, :]
        assert torch.norm(r, p=-1) == 0
    input_ids = torch.tensor([0, 513], dtype=torch.long).to(device)

    embed1 = vocab_embedding_copy(input_ids)
    embed2 = parallel_vocab_embedding(input_ids)

    if rank == 0:
        r = embed1[:, 0:512] - embed2[:, 0:512]
        assert torch.norm(r, p=-1) == 0
    elif rank == 1:
        r = embed1[:, 512:1024] - embed2[:, 512:1024]
        assert torch.norm(r, p=-1) == 0

    res = torch.norm(embed1 - embed2, p=1)
    assert res == 0

    atorch.reset_distributed()


class TestVocabParallelEmbedding(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_run_megatron_vocab_parallel_embedding(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        world_size = 2
        mp.spawn(
            _run_megatron_vocab_parallel_embedding,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestVocabParallelCrossEntropy(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_run_megatron_vocab_parallel_embedding(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        world_size = 2
        mp.spawn(
            _run_megatron_vocab_parallel_cross_entropy,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestParallelEmbedding(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_run_megatron_parallel_embedding(self):

        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        world_size = 2
        mp.spawn(
            _run_megatron_parallel_embedding,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
