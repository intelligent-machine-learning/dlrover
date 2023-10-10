#!/usr/bin/env python
# coding=utf-8
import copy
import os
import unittest

import torch
import torch.multiprocessing as mp
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPAttention

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import create_parallel_group
from atorch.modules.distributed_modules.transformer import MegatronBertAttention, MegatronCLIPAttention

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)

    atorch.init_distributed("nccl")
    torch.cuda.device(atorch.local_rank())
    parallel_config = ([("model", world_size)], None)

    create_parallel_group(parallel_config)


def _run_megatron_bert_attn(rank, world_size):
    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    device = torch.device("cuda:{}".format(atorch.local_rank()))
    torch.cuda.set_device(device)

    bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "vocab_size": 30522,
    }
    config = BertConfig(**bert_config)
    torch.manual_seed(1)

    bert_attn = BertAttention(config=config)
    bert_attn_copy = copy.deepcopy(bert_attn)
    megatron_attn = MegatronBertAttention(orig_module=bert_attn, process_group=pg, ranks=ranks, defer_init=False)

    input_ = torch.ones((2, 8, 1024)).to(device)
    bert_attn_copy.to(device)
    megatron_attn.to(device)
    input_ = input_.to(device)
    bert_attn_copy.eval()
    megatron_attn.eval()
    parallel_out = megatron_attn(input_)
    out = bert_attn_copy(input_)
    atorch.reset_distributed()
    assert torch.norm(parallel_out[0] - out[0], p=-1) == 0


def _run_megatron_clip_attn(rank, world_size):
    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    device = torch.device("cuda:{}".format(atorch.local_rank()))
    torch.cuda.set_device(device)
    clip_config = {
        "hidden_size": 1024,
    }
    config = CLIPTextConfig(**clip_config)
    torch.manual_seed(1)

    clip_attn = CLIPAttention(config=config)
    print(config)
    ranks = [0, 1]
    clip_attn_copy = copy.deepcopy(clip_attn)
    megatron_clip_attn = MegatronCLIPAttention(orig_module=clip_attn, process_group=pg, ranks=ranks, defer_init=False)
    input_ = torch.ones((2, 16, 1024))
    clip_attn_copy.to(device)
    megatron_clip_attn.to(device)
    input_ = input_.to(device)
    clip_attn_copy.eval()
    megatron_clip_attn.eval()
    parallel_out, _ = megatron_clip_attn(input_)
    out, _ = clip_attn_copy(input_)
    assert torch.norm(parallel_out - out, p=-1) == 0
    atorch.reset_distributed()


class TestMegatronOperator(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_megatron_bert_attn(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        world_size = 2
        mp.spawn(
            _run_megatron_bert_attn,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_megatron_clip_attn(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _run_megatron_clip_attn,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
