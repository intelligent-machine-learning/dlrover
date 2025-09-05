import os
import time
import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper
)
from contextlib import nullcontext


def human_readable_flops(num):
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.3f%s" % (num, unit)
        num /= 1000.0
    return "%.3f%s" % (num, "Yi")


def compute_training_flops(
    batch_size,
    sequence_length,
    hidden_size,
    vocab_size,
    intermediate_size,
    num_layers,
    use_gradient_checkpointing=False,
    use_peft=False,
    use_gqa=False,
    kv_head_ratio=1,
):
    """Returns:
    hardware flops
    model flops

    The source of formula:
    Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM's
    (APPENDIX: FLOATING-POINT OPERATIONS)

    Assuming that backward pass has twice FLOPs as many as forward pass. Only matrix multiplication FLOPs are computed.
    For use_peft, backward pass FLOPS is a little more than the forward pass. Assuming equal for simplicity here.
    """
    # [b,s,n] -> [b,s,n]
    query_proj_flops = batch_size * 2 * sequence_length * hidden_size**2
    if use_gqa:
        key_value_proj_flops = (
            2
            * batch_size
            * 2
            * sequence_length
            * hidden_size
            * hidden_size
            / kv_head_ratio
        )
    else:
        key_value_proj_flops = 2 * query_proj_flops
    attention_proj_flops = query_proj_flops + key_value_proj_flops
    attention_flops = (
        2 * batch_size * hidden_size * sequence_length**2
        + 4 * batch_size * sequence_length * hidden_size**2
    )
    attention_forward_flops = attention_proj_flops + attention_flops
    # llama2 use gate_proj, has 3 Linears
    two_mlps_forward_flops = (
        3 * 2 * batch_size * sequence_length * hidden_size * intermediate_size
    )
    logits_forward_flops = 2 * batch_size * sequence_length * hidden_size * vocab_size
    decoder_layer_forward_flops = attention_forward_flops + two_mlps_forward_flops
    # forward FLOPs without gradient checkpointing
    forward_flops_wo_gc = (
        num_layers * decoder_layer_forward_flops + logits_forward_flops
    )
    factor = 2 if use_peft else 3
    if not use_gradient_checkpointing:
        return forward_flops_wo_gc * factor, forward_flops_wo_gc * factor
    else:
        return (
            num_layers * decoder_layer_forward_flops * (factor + 1)
            + logits_forward_flops * factor,
            forward_flops_wo_gc * factor,
        )




def apply_fsdp_checkpointing(model, blocks):
    wrapper = lambda m: checkpoint_wrapper(m,
        checkpoint_fn=torch.utils.checkpoint.checkpoint,
        use_reentrant=False,
        preserve_rng_state=True)
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


class DummyDataset(Dataset):
    def __init__(self, vocab_size=1000, max_length=128, data_size=100000):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        text = torch.randint(low=0, high=self.vocab_size, size=(self.max_length,))
        return text, text


def main():
    # Initialize the process group
    dist.init_process_group(backend="nccl")

    # Get local rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    num_layers = 10
    hidden_size = 4096
    intermediate_size = 8192
    vocab_size = 126464
    num_head = 64
    num_kv_head = 8
    batch_size = 2
    seq_length = 4096
    kv_head_ratio = num_head // num_kv_head
    torch.cuda.set_device(local_rank)


    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_head,
        num_key_value_heads=num_kv_head,
        intermediate_size=intermediate_size,
        max_position_embeddings=seq_length,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
    #    attn_implementation="flash_attention_2",
        use_cache=False,
        use_bfloat16=True
    )

    #init_device = "cpu" if local_rank == 0 else "meta"
    init_device = "meta"

    # from liger_kernel.transformers import apply_liger_kernel_to_llama
    # apply_liger_kernel_to_llama(
    #   rope=True,
    #   swiglu=True,
    #   cross_entropy=True,
    #   fused_linear_cross_entropy=False,
    #   rms_norm=True
    # )

    with torch.device(init_device):
        model = LlamaForCausalLM(config)

    
    flop, _ = compute_training_flops(
        batch_size,
        seq_length,
        hidden_size,
        vocab_size,
        intermediate_size,
        num_layers,
        use_gradient_checkpointing=True,
        use_gqa=True,
        kv_head_ratio=kv_head_ratio,
    )


    dataset = DummyDataset(vocab_size=vocab_size, max_length=seq_length)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    #param_init_fn = lambda m: m.to_empty(device=torch.device("cuda")) if local_rank != 0 else None
    param_init_fn = lambda m: m.to_empty(device=torch.device("cuda"))
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer,},)
    #model = model.to(dtype=torch.bfloat16)

    model = FSDP(model, device_id=local_rank, auto_wrap_policy=wrap_policy,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
                 sync_module_states=False, param_init_fn=param_init_fn,
                 forward_prefetch=True, limit_all_gathers=True, use_orig_params=True)

    apply_fsdp_checkpointing(model, LlamaDecoderLayer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training Loop
    def save_profile(prof):
        prof.export_chrome_trace(f"fsdp_trace_{rank}.json")

    epoch = 0
    iters = 0
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1000000,
            repeat=1),
        on_trace_ready=save_profile,
        record_shapes=False,
        with_stack=False)

    #prof = nullcontext()
    # prof.start()

    dur = []
    model.train()
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(local_rank), labels.to(local_rank)
        start = time.time()
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if rank == 0:
            dur = time.time() - start
            tflops = flop / dur / 1e12
            print(f"Epoch {epoch}, Loss: {loss.item()} time {dur} tflops {tflops}")
        iters += 1
        # if iters > 10:
        #     break
        # prof.step()
    # prof.stop()

    print("Training Complete")
    dist.destroy_process_group()

def main_ds():
    import deepspeed
    dist.init_process_group(backend="nccl")

    # Get local rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    num_layers = 20
    hidden_size = 8192//4
    intermediate_size = 28672
    vocab_size = 126464
    num_head = 64
    num_kv_head = 8
    batch_size = 2
    seq_length = 4096
    kv_head_ratio = num_head // num_kv_head
    torch.cuda.set_device(local_rank)

    flop, _ = compute_training_flops(
        batch_size,
        seq_length,
        hidden_size,
        vocab_size,
        intermediate_size,
        num_layers,
        use_gradient_checkpointing=True,
        use_gqa=True,
        kv_head_ratio=kv_head_ratio,
    )


    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_head,
        num_key_value_heads=num_kv_head,
        intermediate_size=intermediate_size,
        max_position_embeddings=seq_length,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        attn_implementation="flash_attention_2",
        use_cache=False,
        use_bfloat16=True
    )

    #init_device = "cpu" if local_rank == 0 else "meta"
    init_device = "meta"

    from liger_kernel.transformers import apply_liger_kernel_to_llama
    apply_liger_kernel_to_llama(
      rope=True,
      swiglu=True,
      cross_entropy=True,
      fused_linear_cross_entropy=False,
      rms_norm=True
    )

    ds_config = {
        "train_batch_size": batch_size * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
        #"steps_per_print": 10,
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
        },
        "bf16": {
            "enabled": True,
        },
        "activation_checkpointing": {
            "partition_activations": True,            # Partition activations across GPUs
            #"contiguous_memory_optimization": True,   # Optimize contiguous memory usage
        },
    }

    kwargs = {}
    kwargs["config"] = ds_config
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = LlamaForCausalLM(config)
    kwargs["model"] = model

    from deepspeed.ops.adam import FusedAdam
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = FusedAdam(model.parameters(), lr=1e-4)
    kwargs["optimizer"] = optimizer
    model_engine, optimizer, _, _ = deepspeed.initialize(**kwargs)
    #from remote_pdb import RemotePdb
    #RemotePdb("127.0.0.1", 16666+rank).set_trace()

    dataset = DummyDataset(vocab_size=vocab_size, max_length=seq_length)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    model_engine.train()
    start = end = 0
    dur = []
    epoch = 0
    # Training Loop
    def save_profile(prof):
        prof.export_chrome_trace(f"ds_trace_{rank}.json")

    epoch = 0
    iters = 0
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=3,
    #         repeat=1),
    #     on_trace_ready=save_profile,
    #     record_shapes=True,
    #     with_stack=True)

    # #prof = nullcontext()
    # prof.start()

    for step, (input_ids, labels) in enumerate(dataloader):
        start = time.time()
        input_ids, labels = input_ids.to(local_rank), labels.to(local_rank)
        optimizer.zero_grad()
        loss = model_engine(input_ids=input_ids, labels=labels).loss
        model_engine.backward(loss)
        model_engine.step()
        torch.cuda.synchronize()
        dur = time.time() - start
        tflops = flop / dur / 1e12
        if rank == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {loss.item()}, time {dur}, {tflops}")
        # if step > 10:
        #     break
        # prof.step()
    # prof.stop()

def main_qwen_vl():
    #dist.init_process_group(backend="nccl")

    #local_rank = int(os.environ["LOCAL_RANK"])
    #rank = int(os.environ["RANK"])
    #world_size = int(os.environ["WORLD_SIZE"])
    world_size = 1
    local_rank = rank = 0

    torch.cuda.set_device(local_rank)
    config ={
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "vision_start_token_id": 151652,
      "vision_end_token_id": 151653,
      "vision_token_id": 151654,
      "image_token_id": 151655,
      "video_token_id": 151656,
      "hidden_act": "silu",
      "hidden_size": 8192 // 4,
      "initializer_range": 0.02,
      "intermediate_size": 29568 // 4,
      "max_position_embeddings": 32768,
      "max_window_layers": 80,
      "model_type": "qwen2_vl",
      "num_attention_heads": 64,
      "num_hidden_layers": 1,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": False,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.41.2",
      "use_cache": False,
      "use_sliding_window": False,
      "vision_config": {
        "depth": 32,
        "embed_dim": 1280,
        "mlp_ratio": 4,
        "num_heads": 16,
        "in_chans": 3,
        "hidden_size": 8192,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "temporal_patch_size": 2
      },
      "rope_scaling": {
        "type": "mrope",
        "mrope_section": [
          16 // 4,
          24 // 4,
          24 // 4
        ]
      },
      "vocab_size": 152064
    } 

    from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel, Qwen2VisionTransformerPretrainedModel, Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    qwen_config = Qwen2VLConfig(**config)
    preprocess_config = {
      "min_pixels": 3136,
      "max_pixels": 12845056,
      "patch_size": 14,
      "temporal_patch_size": 2,
      "merge_size": 2,
      "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "image_processor_type": "Qwen2VLImageProcessor",
      "processor_class": "Qwen2VLProcessor"
    }
    preprocess = Qwen2VLImageProcessor(**preprocess_config)
    with torch.device('cuda'):
        #model = Qwen2VLForConditionalGeneration(qwen_config)
        text_model = Qwen2VLModel(qwen_config)
        vision_model = Qwen2VisionTransformerPretrainedModel(qwen_config.vision_config)
        image = [torch.ones(1280, 1280, 3, dtype=torch.uint8) for _ in range(10)]
    if rank == 0:
        text = torch.randint(low=0, high=qwen_config.vocab_size, size=(1, 4096,)).cuda()
        #text[-32:] = 1516545
        #print((text == 151655).sum().item())
        data = preprocess(image, return_tensors="pt")
        image_grid_thw = data['image_grid_thw'].cuda()
        image_hidden = data['pixel_values'].cuda()
        #
        t = text_model(text)
        v = vision_model(image_hidden, image_grid_thw)
        breakpoint()
        #m = model(input_ids=text,pixel_values=image_hidden,image_grid_thw=image_grid_thw)
        print(1)

    #num_layers = 10
    #hidden_size = 8192
    #intermediate_size = 32768
    #vocab_size = 126464
    #num_head = 128
    #num_kv_head = 16
    #batch_size = 2
    #seq_length = 4096
    #kv_head_ratio = num_head // num_kv_head
    #torch.cuda.set_device(local_rank)


    #config = LlamaConfig(
    #    vocab_size=vocab_size,
    #    hidden_size=hidden_size,
    #    num_hidden_layers=num_layers,
    #    num_attention_heads=num_head,
    #    num_key_value_heads=num_kv_head,
    #    intermediate_size=intermediate_size,
    #    max_position_embeddings=seq_length,
    #    initializer_range=0.02,
    #    layer_norm_eps=1e-5,
    #    attn_implementation="flash_attention_2",
    #    use_cache=False,
    #    use_bfloat16=True
    #)

    ##init_device = "cpu" if local_rank == 0 else "meta"
    #init_device = "meta"

    #from liger_kernel.transformers import apply_liger_kernel_to_llama
    #apply_liger_kernel_to_llama(
    #  rope=True,
    #  swiglu=True,
    #  cross_entropy=True,
    #  fused_linear_cross_entropy=False,
    #  rms_norm=True
    #)

    #with torch.device(init_device):
    #    model = LlamaForCausalLM(config)

    #
    #flop, _ = compute_training_flops(
    #    batch_size,
    #    seq_length,
    #    hidden_size,
    #    vocab_size,
    #    intermediate_size,
    #    num_layers,
    #    use_gradient_checkpointing=True,
    #    use_gqa=True,
    #    kv_head_ratio=kv_head_ratio,
    #)


    #dataset = DummyDataset(vocab_size=vocab_size, max_length=seq_length)
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    ##param_init_fn = lambda m: m.to_empty(device=torch.device("cuda")) if local_rank != 0 else None
    #param_init_fn = lambda m: m.to_empty(device=torch.device("cuda"))
    #wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer,},)
    ##model = model.to(dtype=torch.bfloat16)

    #model = FSDP(model, device_id=local_rank, auto_wrap_policy=wrap_policy,
    #             mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
    #             sync_module_states=False, param_init_fn=param_init_fn,
    #             forward_prefetch=True, limit_all_gathers=True, use_orig_params=True)

    #apply_fsdp_checkpointing(model, LlamaDecoderLayer)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    ## Training Loop
    #def save_profile(prof):
    #    prof.export_chrome_trace(f"fsdp_trace_{rank}.json")

    #epoch = 0
    #iters = 0
    #prof = torch.profiler.profile(
    #    schedule=torch.profiler.schedule(
    #        wait=1,
    #        warmup=1,
    #        active=3,
    #        repeat=1),
    #    on_trace_ready=save_profile,
    #    record_shapes=True,
    #    with_stack=True)

    ##prof = nullcontext()
    ##prof.start()

    #dur = []
    #with prof:
    #    model.train()
    #    for input_ids, labels in dataloader:
    #        start = time.time()
    #        input_ids, labels = input_ids.to(local_rank), labels.to(local_rank)
    #        optimizer.zero_grad()
    #        loss = model(input_ids=input_ids, labels=labels).loss
    #        loss.backward()
    #        optimizer.step()
    #        torch.cuda.synchronize()
    #        if rank == 0:
    #            dur = time.time() - start
    #            tflops = flop / dur / 1e12
    #            print(f"Epoch {epoch}, Loss: {loss.item()} time {dur} tflops {tflops}")
    #        iters += 1
    #        if iters > 10:
    #            break
    #        prof.step()
    #    epoch += 1

    #print("Training Complete")
    #dist.destroy_process_group()
def mllama():
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig, MllamaTextConfig
    from transformers.models.mllama.modeling_mllama import MllamaVisionEncoderLayer, MllamaSelfAttentionDecoderLayer
    from liger_kernel.transformers import apply_liger_kernel_to_mllama

    dist.init_process_group(backend="nccl")

    # Get local rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    apply_liger_kernel_to_mllama(
      rope=True,
      swiglu=True,
      cross_entropy=False,
      fused_linear_cross_entropy=True,
      rms_norm=True
    )
    config = {
      "architectures": [
        "MllamaForConditionalGeneration"
      ],
      "image_token_index": 128256,
      "model_type": "mllama",
      "text_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": 128000,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "cross_attention_layers": [
          3,
          8,
          13,
          18,
          23,
          28,
          33,
          38,
          43,
          48,
          53,
          58,
          63,
          68,
          73,
          78,
          83,
          88,
          93,
          98
        ],
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": [
          128001,
          128008,
          128009
        ],
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "id2label": {
          "0": "LABEL_0",
          "1": "LABEL_1"
        },
        "initializer_range": 0.02,
        "intermediate_size": 28672,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
          "LABEL_0": 0,
          "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 131072,
        "min_length": 0,
        "model_type": "mllama_text_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 64,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 40,
        "num_key_value_heads": 8,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 128004,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
          "factor": 8.0,
          "high_freq_factor": 4.0,
          "low_freq_factor": 1.0,
          "original_max_position_embeddings": 8192,
          "rope_type": "llama3"
        },
        "rope_theta": 500000.0,
        "sep_token_id": None,
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": False,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "use_cache": False,
        "vocab_size": 128256
      },
      "torch_dtype": "bfloat16",
      "transformers_version": "4.45.0.dev0",
      "vision_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "attention_heads": 16,
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "bos_token_id": None,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "id2label": {
          "0": "LABEL_0",
          "1": "LABEL_1"
        },
        "image_size": 560,
        "intermediate_layers_indices": [
          3,
          7,
          15,
          23,
          30
        ],
        "intermediate_size": 5120,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {
          "LABEL_0": 0,
          "LABEL_1": 1
        },
        "length_penalty": 1.0,
        "max_length": 20,
        "max_num_tiles": 4,
        "min_length": 0,
        "model_type": "mllama_vision_model",
        "no_repeat_ngram_size": 0,
        "norm_eps": 1e-05,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_channels": 3,
        "num_global_layers": 8,
        "num_hidden_layers": 32,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": None,
        "patch_size": 14,
        "prefix": None,
        "problem_type": None,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "supported_aspect_ratios": [
          [
            1,
            1
          ],
          [
            1,
            2
          ],
          [
            1,
            3
          ],
          [
            1,
            4
          ],
          [
            2,
            1
          ],
          [
            2,
            2
          ],
          [
            3,
            1
          ],
          [
            4,
            1
          ]
        ],
        "suppress_tokens": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tf_legacy_loss": False,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": "bfloat16",
        "torchscript": False,
        "typical_p": 1.0,
        "use_bfloat16": False,
        "vision_output_dim": 7680
      }
    }
    vision_config = MllamaVisionConfig(**config['vision_config'])
    text_config = MllamaTextConfig(**config['text_config'])
    model_config = MllamaConfig(vision_config, text_config, torch_dtype="bfloat16")
    data = torch.load('dummy.pth', map_location='cuda')
    label = torch.randint(low=0, high=config['text_config']['vocab_size'], size=data['input_ids'].shape)
    with torch.device('meta'):
        model = MllamaForConditionalGeneration(model_config)
    param_init_fn = lambda m: m.to_empty(device=torch.device("cuda"))
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={ MllamaVisionEncoderLayer, MllamaSelfAttentionDecoderLayer},)
    #model = model.to(dtype=torch.bfloat16)

    model = FSDP(model, device_id=local_rank, auto_wrap_policy=wrap_policy,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
                 sync_module_states=False, param_init_fn=param_init_fn,
                 forward_prefetch=True, limit_all_gathers=True, use_orig_params=True)

    apply_fsdp_checkpointing(model, (MllamaVisionEncoderLayer, MllamaSelfAttentionDecoderLayer))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    def save_profile(prof):
        prof.export_chrome_trace(f"mllama_trace_{rank}.json")

    epoch = 0
    iters = 0
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=2,
    #         repeat=1),
    #     on_trace_ready=save_profile,
    #     record_shapes=True,
    #     with_stack=True)

    # prof.start()
    model.train()
    for i in range(100000000):
        start = time.time()
        optimizer.zero_grad()
        loss= model(**data, labels=label).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        dur = time.time() - start
        if rank == 0:
            print(f"Step {i}, Loss {loss.item()}, time {dur}")
    #    prof.step()
    # prof.stop()


if __name__ == "__main__":
    # main_ds()
    main()
    # mllama()

