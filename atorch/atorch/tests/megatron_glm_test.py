import copy
import os
import unittest

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import atorch
from atorch.distributed.distributed import (
    ParallelGroupContextManager,
    create_parallel_group,
    parallel_group,
    parallel_group_and_ranks,
    reset_distributed,
)
from atorch.modules.distributed_modules.transformer import (
    MegatronGLMBlock,
    MegatronGLMMLP,
    MegatronGLMModel,
    MegatronGLMSelfAttention,
    MegatronGLMStack,
)
from atorch.tests.glm.modeling_glm import GLMBlock, GLMConfig, GLMModel, GLMStack, SelfAttention


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)

    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")

    torch.cuda.device(atorch.local_rank())


def _run_megatron_glm_stack(rank, world_size):
    init_dist(rank, world_size)
    create_parallel_group(([("tensor", 2)], None))
    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")
    # layers
    # hidden_size
    # num_attention_heads
    # attention_dropout_prob
    # output_dropout_prob
    # layernorm_epsilon
    # init_method
    # output_layer_init_method

    config = [2, 1024, 8, 200, 0.5, 0.5, 0.5, None]
    torch.manual_seed(1)
    glm_stack = GLMStack(*config)
    ranks = [0, 1]
    glm_stack_copy = copy.deepcopy(glm_stack)
    pg, ranks = parallel_group_and_ranks("tensor")
    megatron_glm_stack = MegatronGLMStack(
        *config, orig_module=glm_stack, process_group=pg, ranks=ranks, defer_init=False
    )
    input_ = torch.ones((2, 16, 1024), dtype=torch.long).to(device)
    attention_masks = torch.ones((2, 1, 16, 16), dtype=torch.long).to(device)
    position_ids = torch.ones((2, 16), dtype=torch.long).to(device)
    glm_stack_copy.to(device)
    megatron_glm_stack.to(device)
    input_ = input_.to(device)
    glm_stack_copy.eval()
    megatron_glm_stack.eval()
    ltor_mask = torch.ones(2, 1, 16, 16).to(device)

    glm_stack_block = glm_stack_copy.layers[0].eval()

    megatron_stack_block = megatron_glm_stack.layers[0].eval()

    if rank == 0:
        r = (
            glm_stack_block.attention.query_key_value.weight[0:512, :]
            - megatron_stack_block.attention.query_key_value.weight[0:512, :]
        )
        assert torch.norm(r, p=-1) == 0
    else:
        r = (
            glm_stack_block.attention.query_key_value.weight[512:1024, :]
            - megatron_stack_block.attention.query_key_value.weight[0:512, :]
        )
        assert torch.norm(r, p=-1) == 0
    hidden_states = torch.ones(2, 16, 1024).to(device)
    res1 = glm_stack_block.input_layernorm(hidden_states)
    res2 = megatron_stack_block.input_layernorm(hidden_states)
    assert torch.norm(res1 - res2, p=-1) == 0

    attention_output1 = glm_stack_block.attention(res1, ltor_mask)
    attention_output2 = megatron_stack_block.attention(res2, ltor_mask)
    assert torch.norm(attention_output1 - attention_output2, p=-1) == 0

    layernorm_input1 = res1 + attention_output1
    layernorm_input2 = res2 + attention_output2

    assert torch.norm(layernorm_input1 - layernorm_input2, p=-1) == 0
    layernorm_output1 = glm_stack_block.post_attention_layernorm(layernorm_input1)
    layernorm_output2 = megatron_stack_block.post_attention_layernorm(layernorm_input2)
    assert torch.norm(layernorm_output1 - layernorm_output2, p=-1) == 0

    output1 = glm_stack_block(hidden_states, ltor_mask)
    output2 = megatron_stack_block(hidden_states, ltor_mask)
    assert torch.norm(output1 - output2, p=-1) == 0

    out, _ = glm_stack_copy(input_, position_ids, attention_masks)
    parallel_out, _ = megatron_glm_stack(input_, position_ids, attention_masks)

    assert torch.norm(out - parallel_out, p=-1) == 0

    # assert torch.norm(r, p=-1) == 0
    reset_distributed()


def _run_megatron_glm_block(rank, world_size):
    init_dist(rank, world_size)
    create_parallel_group(([("tensor", 2)], None))
    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    # hidden_size
    # num_attention_heads
    # attention_dropout_prob
    # output_dropout_prob
    # layernorm_epsilon
    # init_method
    # output_layer_init_method
    config = [1024, 8, 0.5, 0.5, 0.5, None, None]
    torch.manual_seed(1)
    glm_block = GLMBlock(*config)
    ranks = [0, 1]
    pg, ranks = parallel_group_and_ranks("tensor")
    glm_block_copy = copy.deepcopy(glm_block)
    megatron_glm_block = MegatronGLMBlock(orig_module=glm_block, process_group=pg, ranks=ranks, defer_init=False)
    input_ = torch.ones((2, 16, 1024)).to(device)
    ltor_mask = torch.ones(2, 1, 16, 16).to(device)
    glm_block_copy.to(device)
    megatron_glm_block.to(device)
    input_ = input_.to(device)
    glm_block_copy.eval()
    megatron_glm_block.eval()

    out = glm_block_copy(input_, ltor_mask)
    parallel_out = megatron_glm_block(input_, ltor_mask)

    r = parallel_out - out
    assert torch.norm(r, p=-1) == 0
    reset_distributed()


def _run_megatron_glm_attn_no_rotary(rank, world_size):
    init_dist(rank, world_size)
    create_parallel_group(([("tensor", 2)], None))
    pg, ranks = parallel_group_and_ranks("tensor")
    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    config = [1024, 8, 0.5, 0.5, None]
    # hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method
    torch.manual_seed(1)
    self_attn = SelfAttention(*config)
    ranks = [0, 1]
    self_attn_copy = copy.deepcopy(self_attn)
    megatron_self_attn = MegatronGLMSelfAttention(
        orig_module=self_attn, process_group=pg, ranks=ranks, defer_init=False
    )
    input_ = torch.rand((2, 16, 1024)).to(device)
    ltor_mask = torch.randint(0, 2, (2, 1, 16, 16)).to(device)
    self_attn_copy.to(device)
    megatron_self_attn.to(device)
    input_ = input_.to(device)
    self_attn_copy.eval()

    megatron_self_attn.eval()
    if rank == 0:
        r = self_attn_copy.query_key_value.weight[0:512, :] - megatron_self_attn.query_key_value.weight[0:512, :]
        assert torch.norm(r, p=2) == 0
        r = (
            self_attn_copy.query_key_value.weight[1024 + 0 : 1024 + 512, :]
            - megatron_self_attn.query_key_value.weight[512 + 0 : 512 + 512, :]
        )
        assert torch.norm(r, p=2) == 0
        r = (
            self_attn_copy.query_key_value.weight[1024 * 2 + 0 : 1024 * 2 + 512, :]
            - megatron_self_attn.query_key_value.weight[512 * 2 + 0 : 512 * 2 + 512, :]
        )
        assert torch.norm(r, p=2) == 0
        r = self_attn_copy.dense.weight[:, 0:512] - megatron_self_attn.dense.weight
        assert torch.norm(r, p=2) == 0
    else:
        r = self_attn_copy.query_key_value.weight[512:1024, :] - megatron_self_attn.query_key_value.weight[0:512, :]
        assert torch.norm(r, p=2) == 0
        r = (
            self_attn_copy.query_key_value.weight[1024 + 512 : 1024 + 1024, :]
            - megatron_self_attn.query_key_value.weight[512 + 0 : 512 + 512, :]
        )
        assert torch.norm(r, p=2) == 0
        r = (
            self_attn_copy.query_key_value.weight[1024 * 2 + 512 : 1024 * 2 + 1024, :]
            - megatron_self_attn.query_key_value.weight[512 * 2 + 0 : 512 * 2 + 512, :]
        )
        assert torch.norm(r, p=2) == 0
        r = self_attn_copy.dense.weight[:, 512:1024] - megatron_self_attn.dense.weight
        assert torch.norm(r, p=2) == 0

    out = self_attn_copy(input_, ltor_mask)
    parallel_out = megatron_self_attn(input_, ltor_mask)

    r = parallel_out - out

    assert r.abs().max() < 5e-6

    ranks = [0, 1]

    input_ = torch.ones((2, 16, 1024)).to(device)
    ltor_mask = torch.ones((2, 1, 16, 16)).to(device)
    out = self_attn_copy.half()(input_.half(), ltor_mask.half())
    parallel_out = megatron_self_attn.half()(input_.half(), ltor_mask.half())
    assert r.abs().max() < 5e-6

    reset_distributed()


def _run_megatron_glm_attn_rotary(rank, world_size):
    init_dist(rank, world_size)
    create_parallel_group(([("tensor", 2)], None))
    pg, ranks = parallel_group_and_ranks("tensor")
    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    config = [1024, 8, 0.5, 0.5, None]
    # hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method
    torch.manual_seed(1)
    self_attn = SelfAttention(*config, use_rotary=True)
    assert self_attn.use_rotary is True
    ranks = [0, 1]
    self_attn_copy = copy.deepcopy(self_attn)

    # input_ = torch.rand((2, 16, 1024)).to(device)
    # ltor_mask = torch.randint(0,2, (2,1,16,16)).to(device)
    input_ = torch.ones((1, 16, 1024)).to(device)
    ltor_mask = torch.ones((1, 1, 16, 16)).to(device)
    position_ids = torch.arange(0, 16, dtype=torch.long, device=device)
    block_position_ids = torch.zeros(16, dtype=torch.long, device=device)
    position_ids = torch.stack((position_ids, block_position_ids), dim=0).unsqueeze(0)

    megatron_self_attn = MegatronGLMSelfAttention(
        orig_module=self_attn,
        process_group=pg,
        ranks=ranks,
        defer_init=False,
        use_rotary=True,
    ).to(device)
    megatron_self_attn.eval()
    self_attn_copy.eval()

    parallel_out = megatron_self_attn(input_, ltor_mask, position_ids=position_ids)

    out = self_attn_copy.to(device)(input_, ltor_mask, position_ids=position_ids)
    res = parallel_out - out
    assert res.abs().max() < 1e-6

    self_attn = SelfAttention(*config, use_rotary=True, use_fa=True)
    assert self_attn.use_fa is True
    assert self_attn.use_rotary is True
    self_attn_copy = copy.deepcopy(self_attn)

    megatron_self_attn = MegatronGLMSelfAttention(
        orig_module=self_attn,
        process_group=pg,
        ranks=ranks,
        defer_init=False,
        use_fa=True,
        use_rotary=True,
    ).to(device)
    assert megatron_self_attn.use_fa is True
    assert megatron_self_attn.use_rotary is True
    megatron_self_attn.eval()
    self_attn_copy.eval()
    out = self_attn_copy.to(device)(input_, ltor_mask, position_ids=position_ids)
    assert megatron_self_attn.use_rotary is True
    assert megatron_self_attn.use_fa is True
    parallel_out = megatron_self_attn(input_, ltor_mask, position_ids=position_ids)

    res = parallel_out - out
    assert res.abs().max() <= 1e-5
    reset_distributed()


class TestMegatronAttnNoRotary(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() <= 2, "run with gpu_num >=2")
    def test_megatron_glm_attn(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 2
        mp.spawn(
            _run_megatron_glm_attn_no_rotary,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestMegatronAttnRotary(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with cpu or gpu_num >=2")
    def test_megatron_glm_attn(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 2
        mp.spawn(
            _run_megatron_glm_attn_rotary,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestMegatronGLMBlock(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_megatron_glm_block(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 2
        mp.spawn(
            _run_megatron_glm_block,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestMegatronGLMStack(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with cpu or gpu_num >=2")
    def test_megatron_glm_stack(self):
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 2
        mp.spawn(
            _run_megatron_glm_stack,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


def _run_glm_model(rank, world_size):

    init_dist(rank, world_size)
    create_parallel_group(([("tensor", 2)], None))

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")
    config = GLMConfig()
    config.num_layers = 1
    config.output_predict = True
    torch.manual_seed(0)
    glm_model = GLMModel(config).to(device)

    class FakeInput:
        input_ids = torch.ones((4, 10), dtype=torch.long).to(device)
        attention_mask = torch.ones((4, 10)).to(device)

    glm_model_copy = copy.deepcopy(glm_model)
    glm_model_copy.eval()

    h1 = glm_model_copy.word_embeddings(FakeInput.input_ids)
    pg, ranks = parallel_group_and_ranks("tensor")
    megatron_glm_model = MegatronGLMModel(
        glm_model.config, orig_module=glm_model, process_group=pg, ranks=ranks, defer_init=False
    ).to(device)
    assert len(megatron_glm_model.transformer.layers) == config.num_layers
    h2 = megatron_glm_model.word_embeddings(FakeInput.input_ids)
    rh = h1 - h2
    assert torch.norm(rh, p=1) == 0
    megatron_glm_model.eval()

    input_shape = FakeInput.input_ids.shape

    position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
    block_position_ids = torch.zeros(input_shape[-1], dtype=torch.long, device=device)
    position_ids = torch.stack((position_ids, block_position_ids), dim=0).unsqueeze(0)
    attention_mask = torch.ones((input_shape[0], input_shape[1]), dtype=torch.long, device=device)

    r1 = glm_model_copy.transformer(h1, position_ids, attention_mask)

    r2 = megatron_glm_model.transformer(h1, position_ids, attention_mask)

    # r1[1], r2[1] is mems
    assert torch.norm(r1[1][1] - r2[1][1], p=-1) == 0
    assert torch.norm(r1[1][0] - r2[1][0], p=-1) == 0
    # r1[0], r2[0] is last hidden states
    assert torch.norm(r2[0] - r1[0], p=-1) == 0
    assert torch.norm(r1[0] - r2[0], p=-1) == 0

    res2 = megatron_glm_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)
    res1 = glm_model_copy(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)
    assert torch.norm(res1.logits - res2.logits, p=-1) == 0
    assert torch.norm(res1.last_hidden_states - res2.last_hidden_states, p=-1) == 0

    reset_distributed()


def _run_glm_model_fsdp_to_tp(rank, world_size):
    init_dist(rank, world_size)
    torch.manual_seed(1)
    config = GLMConfig()
    config.output_predict = True
    if torch.cuda.is_available():
        device = atorch.local_rank()
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    glm_model = GLMModel(config).to(device)
    create_parallel_group(([("tensor", 2)], None))
    create_parallel_group(([("data", 2)], None))

    fsdp_model = FSDP(glm_model, parallel_group("data")).to(atorch.local_rank())
    fsdp_model.eval()

    class FakeInput:
        input_ids = torch.ones((4, 10), dtype=torch.long).to(device)
        attention_mask = torch.ones((4, 10)).to(device)

    with torch.no_grad():
        res1 = fsdp_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)

    for _ in range(3):
        pg, ranks = parallel_group_and_ranks("tensor")
        with FSDP.summon_full_params(fsdp_model):
            megatron_glm_model = MegatronGLMModel(
                fsdp_model.config,
                orig_module=fsdp_model,
                process_group=pg,
                ranks=ranks,
                defer_init=False,
                orig_module_dst_device="cpu",
            ).to(device)

        class FakeInput:
            input_ids = torch.ones((4, 10), dtype=torch.long).to(device)
            attention_mask = torch.ones((4, 10)).to(device)

        res2 = megatron_glm_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)
    res = torch.norm(res1.last_hidden_states - res2.last_hidden_states, p=-1)
    assert res == 0
    reset_distributed()


def _run_glm_mlp_fsdp_to_hp_hybrid(rank, world_size):
    init_dist(rank, world_size)
    from atorch.tests.glm.modeling_glm import MLP

    glm_model = MLP(4, 0.1, None)
    if torch.cuda.is_available():
        device = atorch.local_rank()
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    create_parallel_group(([("data", 4)], None))

    fsdp_model = FSDP(glm_model, parallel_group("data")).to(atorch.local_rank())
    fsdp_model.eval()
    inputs = torch.ones((4, 4)).to(device)
    prefix_name = "inference"
    with ParallelGroupContextManager(prefix_name):
        # Note: transformation of glm from FSDP to TP must be
        # under the scope of ParallelGroupContextManager(prefix_name)
        create_parallel_group(([("tensor", 2), ("data", 2)], None))
        pg, ranks = parallel_group_and_ranks("tensor")
        with FSDP.summon_full_params(fsdp_model):
            megatron_glm_model = MegatronGLMMLP(
                orig_module=fsdp_model,
                process_group=pg,
                ranks=ranks,
                defer_init=False,
            ).to(device)
            megatron_glm_model.eval()
            res1 = megatron_glm_model(inputs)
    res2 = fsdp_model(inputs)
    res = torch.norm(res1 - res2, p=-1)
    print(res)
    reset_distributed()
    # to be developed
    # assert res == 0


def _run_glm_model_fsdp_to_tp_hybrid(rank, world_size):
    init_dist(rank, world_size)
    torch.manual_seed(1)
    config = GLMConfig()
    config.output_predict = True
    if torch.cuda.is_available():
        device = atorch.local_rank()
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    glm_model = GLMModel(config).to(device)
    prefix_name = "inference"
    with ParallelGroupContextManager(prefix_name):
        create_parallel_group(([("tensor", 2), ("data", 2)], None))
    create_parallel_group(([("data", 4)], None))

    fsdp_model = FSDP(glm_model, parallel_group("data")).to(atorch.local_rank())
    fsdp_model.eval()

    class FakeInput:
        input_ids = torch.ones((4, 10), dtype=torch.long).to(device)
        attention_mask = torch.ones((4, 10)).to(device)

    with torch.no_grad():
        res1 = fsdp_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)

    for _ in range(3):
        with ParallelGroupContextManager(prefix_name):
            pg, ranks = parallel_group_and_ranks("tensor")
            # Note: transformation of glm from FSDP to TP must be
            # under the scope of ParallelGroupContextManager(prefix_name)
            with FSDP.summon_full_params(fsdp_model):
                megatron_glm_model = MegatronGLMModel(
                    fsdp_model.config,
                    orig_module=fsdp_model,
                    process_group=pg,
                    ranks=ranks,
                    defer_init=False,
                    orig_module_dst_device="cpu",
                ).to(device)

            class FakeInput:
                input_ids = torch.ones((4, 10), dtype=torch.long).to(device)
                attention_mask = torch.ones((4, 10)).to(device)

            res2 = megatron_glm_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)
    res = torch.norm(res1.last_hidden_states - res2.last_hidden_states, p=-1)
    assert res == 0
    reset_distributed()


class TestGLMModel(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_run_glm_model(self):

        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 2
        mp.spawn(
            _run_glm_model,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestGLMModelFSDPtoTP(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "gpu_num >=2")
    def test_run_glm_model_fsdp_to_tp(self):

        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 2
        mp.spawn(
            _run_glm_model_fsdp_to_tp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestGLMMLPFSDPtoTPHybrid(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 4, "run with gpu_num >=4")
    def test_run_glm_mlp_fsdp_to_hybrid_tp(self):

        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 4
        mp.spawn(
            _run_glm_mlp_fsdp_to_hp_hybrid,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


class TestGLMModelFSDPtoTPHybrid(unittest.TestCase):
    # to be fixed
    @unittest.skipIf(torch.cuda.device_count() < 4, "run with gpu_num >=4")
    def test_run_glm_model_fsdp_to_hybrid_tp(self):

        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5001"
        world_size = 4
        mp.spawn(
            _run_glm_model_fsdp_to_tp_hybrid,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
