import glob
import os
import unittest

import deepspeed
import torch
import torch.multiprocessing as mp
from torch import nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from atorch.common.util_func import find_free_port
from atorch.rl.ds_hybrid_engine.ds_hook import *  # NOQA
from atorch.rl.ds_hybrid_engine.initialize import initialize
from atorch.tests.utils.test_utils import init_dist


def _run_llama2_deepspeed_vllm_test(
    rank,
    world_size,
    model_path,
    max_new_tokens,
    print_output,
    use_zero_init,
    batch_size_per_gpu,
    inference_tp_size,
    use_deepspeed,
):
    init_dist(rank, world_size)

    model_path = model_path
    max_new_tokens = max_new_tokens
    inference_tp_size = inference_tp_size
    batch_size_per_gpu = batch_size_per_gpu

    config = {
        "train_batch_size": batch_size_per_gpu * world_size,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "steps_per_print": 10,
        "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
        "bf16": {
            "enabled": True,
        },
        "hybrid_engine": {"enabled": True, "inference_tp_size": inference_tp_size, "max_out_tokens": 500},
        "gradient_clipping": 1.0,
    }
    kwargs = {}
    kwargs["config"] = config
    device = int(os.environ.get("LOCAL_RANK", 0))

    if use_zero_init:
        with deepspeed.zero.Init(config_dict_or_path=config):
            configuration = LlamaConfig.from_pretrained(model_path, trust_remote_code=True)
            model = LlamaForCausalLM(configuration)

            for p in model.parameters():
                p.requires_grad = False

            trainable_module = model.model.layers[-2:]
            for p in trainable_module.parameters():
                p.requires_grad = True

        # to do check keys values
        new_state_dict = {}
        if deepspeed.comm.get_rank() == 0:
            state_dict = {}
            bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
            for f in bin_files:
                s = torch.load(f, map_location="cpu")
                state_dict.update(s)
            new_state_dict = state_dict
        else:
            state_dict = None

        state_dict = new_state_dict

        def load(module: nn.Module, prefix=""):
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again

            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    module._load_from_state_dict(state_dict, prefix, {}, True, [], [], [])

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")

    else:
        model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    kwargs["model"] = model

    if use_deepspeed:
        m, _, _, _ = initialize(**kwargs)  # type: ignore
        print("model is {}".format(m))
    else:
        m = model.bfloat16().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    m.eval()

    prompt = "介绍下支付宝?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = m.generate(inputs.input_ids.to(device), max_length=max_new_tokens)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = m.generate(inputs.input_ids.to(device), max_length=max_new_tokens)

    # params_list = [p for p in m.parameters() if p.requires_grad is True]
    # params_name_list = [p for p, q in m.named_parameters() if q.requires_grad is True]

    from atorch.rl.inference_backend.vllm_backend import VLLMBackend, VLLMComm

    m.vllm_backend = "s"

    # m.sync_weight()

    vllm_inference_backend = VLLMBackend(checkpoint_path=model_path, gen_kwargs={"max_new_tokens": 500})

    vllm_config = {
        "unfrozen_layers": [31, 30],
        "redis_address": {"ip": "0.0.0.0", "port": 6379},
        "vllm_server_address": {"ip": "0.0.0.0", "port": 5005},
    }
    vllm_comm = VLLMComm(
        vllm_config["vllm_server_address"]["ip"], vllm_config["vllm_server_address"]["port"]
    )  # noqa: F841
    assert vllm_comm is not None
    # except Exception as e:
    with deepspeed.zero.GatheredParameters(list(m.module.parameters(recurse=True))):
        vllm_inference_backend.set_train_model_weights(m.module)
        res = vllm_inference_backend.generate(["介绍下支付宝?"])
        print(res)


@unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
class TestLoadDSModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deepspeed.runtime.utils.see_memory_usage("pre test", force=True)

    def test_load_ds_model_use_zero_init_tp_size_1(self):
        world_size = 2
        model_path = "/mnt1/xuantai.hxd/test/llama2_small"
        max_new_tokens = 500
        print_output = True
        use_zero_init = True
        batch_size_per_gpu = 1
        inference_tp_size = 1
        use_deepspeed = True
        use_zero_init = True

        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _run_llama2_deepspeed_vllm_test,
            args=(
                world_size,
                model_path,
                max_new_tokens,
                print_output,
                use_zero_init,
                batch_size_per_gpu,
                inference_tp_size,
                use_deepspeed,
            ),
            nprocs=world_size,
            join=True,
        )

    def test_load_ds_model_tp_size_2(self):
        world_size = 2

        model_path = "/mnt1/xuantai.hxd/test/llama2_small"
        max_new_tokens = 500
        print_output = True
        use_zero_init = True
        batch_size_per_gpu = 1
        inference_tp_size = 2
        use_deepspeed = True
        use_zero_init = False
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _run_llama2_deepspeed_vllm_test,
            args=(
                world_size,
                model_path,
                max_new_tokens,
                print_output,
                use_zero_init,
                batch_size_per_gpu,
                inference_tp_size,
                use_deepspeed,
            ),
            nprocs=world_size,
            join=True,
        )

    def test_load_ds_model_use_zero_init_tp_size_2(self):
        world_size = 2

        model_path = "/mnt1/xuantai.hxd/test/llama2_small"
        max_new_tokens = 500
        print_output = True
        use_zero_init = True
        batch_size_per_gpu = 1
        inference_tp_size = 2
        use_deepspeed = True
        use_zero_init = True
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _run_llama2_deepspeed_vllm_test,
            args=(
                world_size,
                model_path,
                max_new_tokens,
                print_output,
                use_zero_init,
                batch_size_per_gpu,
                inference_tp_size,
                use_deepspeed,
            ),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
