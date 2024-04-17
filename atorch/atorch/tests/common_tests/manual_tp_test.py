import os
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import create_parallel_group, parallel_group
from atorch.modules.distributed_modules.cross_entropy import vocab_parallel_cross_entropy
from atorch.modules.distributed_modules.layers import _initialize_affine_weight
from atorch.modules.distributed_modules.mappings import copy_to_group
from atorch.modules.distributed_modules.randomizer import init_randomizer
from atorch.utils.manual_tp_utils import TPInfo, hf_init_weights_custom_fn, tp_manual_shard_custom_fn
from atorch.utils.meta_model_utils import build_recorded_module, record_module_init
from atorch.utils.version import torch_version

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def _weight_align(tp_model, ref_model):
    sd = ref_model.state_dict()
    new_sd = dict()
    for k, tensor in sd.items():
        if k.endswith("attn.c_attn.weight"):
            new_tensor = torch.empty(tensor.shape[1] // 2, tensor.shape[0], device=tensor.device)
            _initialize_affine_weight(new_tensor, tensor.shape[1] // 2, 0, stride=3, master_weight=tensor.t())
            new_sd[k] = new_tensor
        elif k.endswith("attn.c_attn.bias"):
            new_tensor = torch.empty(tensor.shape[0] // 2, device=tensor.device)
            _initialize_affine_weight(new_tensor, tensor.shape[0] // 2, 0, stride=3, master_weight=tensor)
            new_sd[k] = new_tensor
        elif k.endswith("mlp.c_fc.bias"):
            new_tensor = torch.empty(tensor.shape[0] // 2, device=tensor.device)
            _initialize_affine_weight(new_tensor, tensor.shape[0] // 2, 0, master_weight=tensor)
            new_sd[k] = new_tensor
        elif k.endswith("mlp.c_fc.weight"):
            new_tensor = torch.empty(tensor.shape[1] // 2, tensor.shape[0], device=tensor.device)
            _initialize_affine_weight(new_tensor, tensor.shape[1] // 2, 0, master_weight=tensor.t())
            new_sd[k] = new_tensor
        elif k.endswith("attn.c_proj.weight") or k.endswith("mlp.c_proj.weight"):
            new_tensor = torch.empty(tensor.shape[1], tensor.shape[0] // 2, device=tensor.device)
            _initialize_affine_weight(new_tensor, tensor.shape[0] // 2, 1, master_weight=tensor.t())
            new_sd[k] = new_tensor
        elif k.endswith("wte.weight"):
            new_tensor = torch.empty(tensor.shape[0] // 2, tensor.shape[1], device=tensor.device)
            _initialize_affine_weight(new_tensor, tensor.shape[0] // 2, 0, master_weight=tensor)
            new_sd[k] = new_tensor
        else:
            new_sd[k] = tensor
    tp_model.load_state_dict(new_sd)


def get_gpt2_tpinfo():
    gpt2_tpinfo = TPInfo()
    gpt2_tpinfo.shard_col({"attn.c_attn": {"stride": 3}}, "mlp.c_fc")
    gpt2_tpinfo.shard_row("attn.c_proj", "mlp.c_proj")
    gpt2_tpinfo.shard_vocab("wte")
    gpt2_tpinfo.replic_drop("resid_dropout", "mlp.dropout", "drop")
    gpt2_tpinfo.parallel_drop("attn_dropout")
    gpt2_tpinfo.shrink({".attn": {"embed_dim", "split_size", "num_heads"}})
    return gpt2_tpinfo


def _run_manual_tp(rank):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "4"
    res = atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    create_parallel_group(([("tensor", 2), ("data", 2)], None))
    init_randomizer()
    device = torch.cuda.current_device()
    gpt2_config = GPT2Config(
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        n_embd=256,
        n_head=4,
        n_layer=3,
        n_positions=512,
        vocab_size=50000,
    )

    # tp model
    with record_module_init():
        meta_model = GPT2Model(gpt2_config)
    hf_init_weights_custom_fn(meta_model)
    tp_manual_shard_custom_fn(meta_model, get_gpt2_tpinfo())
    tp_model = build_recorded_module(meta_model).to(device)

    # ref model
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    ref_model = GPT2Model(gpt2_config).to(device)
    _weight_align(tp_model, ref_model)

    # compute
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    input_ids = torch.randint(50, 40000, (2, 512), device=device)
    position_ids = torch.arange(0, 512, dtype=torch.long, device=device)
    attention_mask = torch.randint(0, 2, (2, 512), device=device)
    labels = torch.randint(50, 40000, (2, 512), device=device)
    loss_mask = torch.randint(0, 2, (2, 512), device=device)

    # out
    ref_out = ref_model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask).last_hidden_state
    tp_out = tp_model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask).last_hidden_state

    # ref loss
    ref_logits = F.linear(ref_out, ref_model.wte.weight)
    ref_losses = torch.nn.CrossEntropyLoss(reduction="none")(ref_logits.view(-1, ref_logits.size(-1)), labels.view(-1))
    ref_loss = torch.sum(ref_losses * loss_mask.view(-1))
    if loss_mask.sum().item() > 0:
        ref_loss = ref_loss / loss_mask.sum()

    # tp loss
    tp_out = copy_to_group(tp_out, group=parallel_group("tensor"))  # VocabParalleEmb need copy last out to group
    tp_logits = F.linear(tp_out, tp_model.wte.weight)
    tp_losses = vocab_parallel_cross_entropy(tp_logits, labels).view(-1)
    tp_loss = torch.sum(tp_losses * loss_mask.view(-1))
    if loss_mask.sum().item() > 0:
        tp_loss = tp_loss / loss_mask.sum()

    # backward
    tp_loss.backward()
    ref_loss.backward()

    # loss compare
    assert torch.all(torch.isclose(tp_out, ref_out, atol=1e-5))

    # grad compare
    for k, param in ref_model.named_parameters():
        ref_grad = param.grad.detach().clone()
        tp_grad = dict(tp_model.named_parameters())[k].grad.detach().clone()
        if k.endswith("attn.c_attn.weight"):
            new_tensor = torch.empty(ref_grad.shape[1] // 2, ref_grad.shape[0], device=ref_grad.device)
            _initialize_affine_weight(new_tensor, ref_grad.shape[1] // 2, 0, stride=3, master_weight=ref_grad.t())
        elif k.endswith("attn.c_attn.bias"):
            new_tensor = torch.empty(ref_grad.shape[0] // 2, device=ref_grad.device)
            _initialize_affine_weight(new_tensor, ref_grad.shape[0] // 2, 0, stride=3, master_weight=ref_grad)
        elif k.endswith("mlp.c_fc.bias"):
            new_tensor = torch.empty(ref_grad.shape[0] // 2, device=ref_grad.device)
            _initialize_affine_weight(new_tensor, ref_grad.shape[0] // 2, 0, master_weight=ref_grad)
        elif k.endswith("mlp.c_fc.weight"):
            new_tensor = torch.empty(ref_grad.shape[1] // 2, ref_grad.shape[0], device=ref_grad.device)
            _initialize_affine_weight(new_tensor, ref_grad.shape[1] // 2, 0, master_weight=ref_grad.t())
        elif k.endswith("attn.c_proj.weight") or k.endswith("mlp.c_proj.weight"):
            new_tensor = torch.empty(ref_grad.shape[1], ref_grad.shape[0] // 2, device=ref_grad.device)
            _initialize_affine_weight(new_tensor, ref_grad.shape[0] // 2, 1, master_weight=ref_grad.t())
        elif k.endswith("wte.weight"):
            new_tensor = torch.empty(ref_grad.shape[0] // 2, ref_grad.shape[1], device=ref_grad.device)
            _initialize_affine_weight(new_tensor, ref_grad.shape[0] // 2, 0, master_weight=ref_grad)
        else:
            new_tensor = ref_grad
        assert torch.all(torch.isclose(new_tensor, tp_grad, atol=1e-5)), f"ref_grad: {new_tensor} \ntp_grad: {tp_grad}"

    atorch.reset_distributed()


class TestManualTP(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() < 4 or torch_version() < (2, 0, 0),  # type: ignore
        "run with cpu or gpu_num >=4, torch 2.0 needed for torch.device context manager.",
    )
    def test_manual_tp(self):

        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _run_manual_tp,
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
