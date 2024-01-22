# Llama2 Pretrain/Finetune


This document presents examples of using ATorch to pretrain or finetune the HuggingFace Llama2 model, including [FSDP (ZeRO3)](#FSDP), 3D hybrid parallelism for [semi-automatic optimization](#DS-3D-Parallel), and [fully automatic optimization](#Automatic-Training-Optimization).

- Note: 
    - Llama2 model and wikitext dataset is used in the examples. If you have already downloaded llama2 model, set environment variable: `MODEL_NAME_OR_PATH=llama2_model_directory`. If you have wikitext in your system, set environment variable: `DATASET_PATH=wikitext_data_directory`

    - If llama2 model or wikitext dataset is not present, the training script will automatically download them for you. Note that downloading may take quite some time.

## FSDP

Fully Sharded Data Parallel (FSDP) is PyTorch's implementation of ZeRO3. This example uses FSDP for distributed training, and can be used with other training optimizations, such as mixed precision (fp16/bf16/fp8), gradient checkpointing, etc. This is implemented by calling auto_accelerate API with load_strategy argument, and load_strategy specifies the training optimization method combination.

### Scripts

- training file [fsdp_llama2.py](fsdp_llama2.py)

- launch script [fsdp_llama2_entry.sh](fsdp_llama2_entry.sh)

```bash
cd dlrover/atorch/examples/llama2
pip install -r requirements.txt

# Configurable environment variable: DATASET_PATH, MODEL_NAME_OR_PATH, PER_DEVICE_TRAIN_BATCH_SIZE, etc.
bash fsdp_llama2_entry.sh

# use fp8
USE_FP8=1 bash fsdp_llama2_entry.sh

# use lora
USE_LORA=1 bash fsdp_llama2_entry.sh
```

Note that transformer_engine is required for fp8. Your can use docker image <code>registry.cn-hangzhou.aliyuncs.com/atorch/atorch:pt210_te</code>, which has transformer_engine pre-installed.

## DS 3D Parallel
### Intro
- For large-scale model training (with 100B+ levels), besides using FSDP/zero3 parallelism, 3D parallelism is widely used in deep learning community. 3D parallelism includes tensor parallel, pipeline parallel, and data parallel. Megatron-LM and DeepSpeed provide excellent 3D parallelism implementation which are popular among users.
- Megatron-LM offers a Col/Row parallel layer that users can incorporate into the model definition to achieve tensor parallelism. DeepSpeed's pipeline parallel feature requires converting the model into a sequential list of LayerSpec, making its usage complicated especially for non-sequential models.

    <details>
    <summary>Megatron-LM embeds ParallelLinear</summary>

    ```python
    class ParallelAttention(MegatronModule):
        def __init__(self, ...):
            ...
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
            ...
            self.dense = mpu.RowParallelLinear(
                projection_size,
                args.hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True)
            ...
    ```

    </details>



    <details>
    <summary>Deepspeed pipeline retrofitting</summary>


    ```python
    def model_provider(pre_process=True, post_process=True):
        ...
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True
            )
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )

    class GPTModelPipe(PipelineModule,MegatronModule):
        def __init__(self, ...):
            ...
            # Embedding layer
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            args.hidden_size,
                                            args.padded_vocab_size,
                                            args.max_position_embeddings,
                                            args.hidden_dropout,
                                            init_method=init_method,
                                            num_tokentypes=num_tokentypes,
                                            tied_weight_attr='word_embeddings_weight'))

            for layer_idx in range(args.num_layers):
                self.specs.append(
                    LayerSpec(ParallelTransformerLayerPipe,
                        init_method=init_method,
                        output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                        args.num_layers),
                        layer_number=layer_idx,
                        self_attn_mask_type=AttnMaskType.causal))
                        
            def _logits_helper(embedding, lm_output):
                """A wrapper to massage inputs/outputs from pipeline. """
                return parallel_lm_logits(
                    lm_output,
                    embedding.word_embeddings_weight,
                    self.parallel_output)

            self.specs.append(
                TiedLayerSpec('embed',
                            EmbeddingPipe,
                            args.hidden_size,
                            args.padded_vocab_size,
                            args.max_position_embeddings,
                            args.hidden_dropout,
                            init_method=init_method,
                            num_tokentypes=num_tokentypes,
                            forward_fn=_logits_helper,
                            tied_weight_attr='word_embeddings_weight')
            )
    ```

    </details>



- ATorch supports 3D parallel training based on DeepSpeed/Megatron, and supports easy usage by using auto_accelerate with ds_3d_parallel optimization method.

### User Interface
- A simple example of using Interface for 3D parallelization of Transformers Model

    <details>
    <summary>example</summary>


    ```python
    from transformers.xxx import XXXConfig, XXXModel

    from atorch.auto.opt_lib.ds_3d_parallel_optimization import DeepSpeed3DParallelConfig
    from atorch.utils.meta_model_utils import record_module_init

    # init distributed environment and create 3d parallel groups
    atorch.init_distributed("nccl")

    # meta model for ds 3d parallel
    with record_module_init():
        meta_model = XXXModel(XXXConfig(...))

    # tensor parallel info and pipeline forward patcher
    ds_3d_parallel_cfg = DeepSpeed3DParallelConfig(
            tpinfo=get_xxx_tpinfo(),
            custom_patcher=get_xxx_custom_patcher(),
        )
    strategy = [
        ("parallel_mode", ([("tensor", tensor_size), ("data", data_size), ("pipeline", pipeline_size)], None)),
        ("deepspeed_3d_parallel", ds_3d_parallel_cfg),
    ]

    # auto_accelerate
    status, result, best_strategy = auto_accelerate(
            meta_model,
            loss_func=my_loss_func,
            load_strategy=strategy,
            ignore_dryrun_on_load_strategy=True,
        )

    # DeepSpeed PipelineEngine model
    model = result.model
    ```
    </details>


- Omits optim and dataset related batch_fn.
- The user specifies the module name information of Tensor Parallel Shard, and the customized forward patcheres if some modules in the pipeline need.

### Related API

- `record_module_init` contextmanager for meta initialization + init args/kwargs recording.

    <details>
    <summary>record_module_init</summary>

    ```python
    # atorch/utils/meta_model_utils.py
    @contextmanager
    def record_module_init():
        """
        Record modules' init args and kwargs while meta constructing model. Since we don't
        save or offload the initial weight, we should reset_paramters or (hf)_init_weights
        after building the real modules with the recorded args/kwargs.
        This contextmanager was originally designed for building deepspeed PipelineModule from
        native torch model implementation.
        """

        def init_record_helper(f):
            @functools.wraps(f)
            def wrapper(module: torch.nn.Module, *args, **kwargs):
                f(module, *args, **kwargs)
                # record args/kwargs after original init, in case parent cls init covers them
                # in mistake; it must be satisfied that args/kwargs not changed in init
                module._init_args = args
                module._init_kwargs = kwargs
                # torch.device('meta') contextmanager may not handle nn.Parameter(...),
                # .to('meta') manually to force everything in meta
                module.to("meta")

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = init_record_helper(cls.__init__)

        def _disable_class(cls):
            cls.__init__ = cls._old_init
            delattr(cls, "_old_init")

        def _init_subclass(cls, **kwargs):
            cls.__init__ = init_record_helper(cls.__init__)

        def substitute_init_recursively(cls, func, visited):
            for subcls in cls.__subclasses__():
                substitute_init_recursively(subcls, func, visited)
                if subcls not in visited:
                    func(subcls)
                    visited.add(subcls)

        try:
            substitute_init_recursively(torch.nn.modules.module.Module, _enable_class, set())
            torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
            torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
            # torch meta init
            torch.device("meta").__enter__()
            yield
        finally:
            substitute_init_recursively(torch.nn.modules.module.Module, _disable_class, set())
            torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass
            delattr(torch.nn.modules.module.Module, "_old_init_subclass")
            torch.device("meta").__exit__()
    ```

    </details>

- `TPInfo` the name information for tensor paralle shard.

    <details>
    <summary>TPInfo</summary>

    ```python
    # atorch/utils/manual_tp_utils.py
    class TPInfo:
        """
        Manual tensor parallel information class.

        Example:
            >>> gpt2_tpinfo = TPInfo()
            >>> gpt2_tpinfo.shard_col({"attn.c_attn": {"stride": 3}}, "mlp.c_fc")
            >>> gpt2_tpinfo.shard_row("attn.c_proj", "mlp.c_proj")
            >>> gpt2_tpinfo.shard_vocab("wte")
            >>> gpt2_tpinfo.replic_drop("resid_dropout", "mlp.dropout", "drop")
            >>> gpt2_tpinfo.parallel_drop("attn_dropout")
            >>> gpt2_tpinfo.shrink({".attn": {"embed_dim", "split_size", "num_heads"}})
            >>> tp_manual_shard_custom_fn(meta_gpt2, gpt2_tpinfo)
        """
        ...
    ```

    </details>

  - Note: `shard_col/row/vocab` supports arguments in `Union[Dict[str, Dict], str]`, where `str` is the suffix of the unique module, the inner `Dict` is the initialization arguments of ATorchTPLayer module (RowParallelLinear /ColumnParallelLinear/ VocabParallelEmbedding).

- `DeepSpeed3DParallelConfig` configuration class

    <details>
    <summary>DeepSpeed3DParallelConfig</summary>

    ```python
    # atorch/auto/opt_lib/ds_3d_parallel_optimization.py
    class DeepSpeed3DParallelConfig:
        def __init__(
            self,
            base_seed=1234,
            tpinfo=None,
            custom_patcher=None,
            tie_first=True,
            logit_helper=None,
            ds_config=None,
            batch_fn=None,
        ):
            self.base_seed = base_seed

            # TPinfo
            self.tpinfo = tpinfo if tpinfo is not None else TPInfo()

            # PipeModuleFromRecordedMeta
            self.custom_patcher = custom_patcher if custom_patcher is not None else dict()
            self.tie_first = tie_first
            # logit helper
            if self.tpinfo.is_vocab_parallelled:
                if logit_helper is None:
                    self.logit_helper = vocab_parallel_logit_helper
                else:
                    logger.warning("Tensor parallel is using VocabParallelEmb, make sure lm_output copied to group")
            else:
                self.logit_helper = logit_helper

            # DeepSpeed config
            self.ds_config = ds_config  # dict() or path

            self.batch_fn = batch_fn if batch_fn is not None else lambda x: x
    ```

    </details>

   - Note: 
      - `base_seed`: for initializing `MultiDimParallelRandomizer`.
      - `tpinfo`: an instance of `TPInfo`.
      - `custom_patcher`: `Optional[Dict[str, patch_fn]]`. `str` is the path_name of module to be patched, from the module_list that `PipeModuleFromRecordedMeta` traverses. `patch_fn` will patch the forward funciton when building the corresponding LayerSpec. GPT2 example:

        <details>
        <summary>gpt2_custom_patcher</summary>

        ```python
        def gpt2_custom_patcher(cfg):
            def wpe_patcher(fw, self):
                @functools.wraps(fw)
                def fw_wrapper(input):
                    assert (
                        isinstance(input, tuple) and len(input) == 3
                    ), "input should be (hidden_states, position_ids, attention_mask)"
                    hidden_states, position_ids, attention_mask = input
                    position_embeddings = fw(position_ids)
                    hidden_states = hidden_states + position_embeddings
                    return hidden_states, attention_mask

                return fw_wrapper

            def h_patcher(fw, self):
                @functools.wraps(fw)
                def fw_wrapper(input):
                    assert isinstance(input, tuple) and len(input) == 2, "input should be (hidden_states, attention_mask)"
                    hidden_states, attention_mask = input
                    ori_attn_mask = attention_mask
                    attention_mask = attention_mask[:, None, None, :]
                    attention_mask = attention_mask.to(hidden_states.dtype)  # fp16 compatibility
                    attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
                    outputs = fw(hidden_states, attention_mask=attention_mask)
                    hidden_states = outputs[0]
                    return hidden_states, ori_attn_mask

                return fw_wrapper

            gpt2_custom_forward_patchers = {"wpe": wpe_patcher}
            gpt2_custom_forward_patchers.update({f"h.{i}": h_patcher for i in range(cfg.n_layer)})
            return gpt2_custom_forward_patchers

            # Note: DeepSpeed needs to require_grad the float tensor passed in the middle, GPT2 h patcher converts the mask, and then returns the original int tensor mask in output.
        ```

        </details>

        

      - `tie_first`: whether to tie the first module in deepspeed pipeline (e.g. vocab embedding in transformer). 
      - `logit_helper`: take effect when `tie_first` is True. The helper function to compute logti on the tied embedding. `_default_logit_helper` will be used if None.
      - `ds_config`: `Dict` for deepspeed config or `str` for json file path. 
      - `batch_fn`:  input adaptation function for deepspeed pipeline (inputs,), (labels,). 

### Scripts

- training file [ds_3d_llama2.py](ds_3d_llama2.py)

- launch script [ds_3d_llama2_entry.sh](ds_3d_llama2_entry.sh)

```bash
cd dlrover/atorch/examples/llama2
pip install -r requirements.txt

# Configurable environment variable: 
# DATASET_PATH, MODEL_NAME_OR_PATH, PIPELINE_PARALLEL_SIZE, MODEL_PARALLEL_SIZE, etc.
# e.x. in a 8-gpu system, to run mp-2 dp-2 pp-2, 
# use `PIPELINE_PARALLEL_SIZE=2 MODEL_PARALLEL_SIZE=2 bash ds_3d_llama2_entry.sh`
bash ds_3d_llama2_entry.sh
```

### Performance
- Model: Llama2-7b
- Cluster: 8 A100-80GB GPUs in each node, with NVLink and 800 Gb/s RDMA

|       | Use LoRA | Num of GPUs | TP | PP | DP | Batch Per GPU | Grad Accu Steps | Global Batch Size | TFLOPs |HFU(%)| Mem Max allocated(MB) |
|-------|----------|------------:|---:|---:|---:|--------------:|----------------:|------------------:|-------:|------|-----------------------|
|  FSDP | Yes      |           8 | /  | /  |  8 |             4 |               1 |                32 | 177.90 | 57.0 | 33354.1               |
| FSDP  | No       |           8 | /  | /  |  8 |             4 |               1 |                32 | 204.67 | 65.6 | 39596.3               |
|  FSDP | Yes      |          16 | /  | /  | 16 |             4 |               1 |                64 | 165.03 | 52.9 | 31735.5               |
| FSDP  | No       |          16 | /  | /  | 16 |             4 |               1 |                64 | 197.22 | 63.2 | 34776.7               |
|  FSDP | Yes      |          32 | /  | /  | 32 |             4 |               1 |               128 | 164.04 | 52.6 | 30926.2               |
| FSDP  | No       |          32 | /  | /  | 32 |             4 |               1 |               128 | 196.04 | 62.8 | 32366.8               |
|  FSDP | Yes      |          64 | /  | /  | 64 |             4 |               1 |               256 | 160.69 | 51.5 | 30521.6               |
| FSDP  | No       |          64 | /  | /  | 64 |             4 |               1 |               256 | 195.14 | 62.5 | 31161.9               |
| DS 3D | /        |           8 |  1 |  2 |  4 |             8 |              64 |              2048 | 190.27 | 61.0 | 40006.9               |
| DS 3D | /        |           8 |  2 |  1 |  4 |             8 |              64 |              2048 | 159.19 | 51.0 | 38635.6               |
| DS 3D | /        |           8 |  2 |  2 |  2 |             8 |              64 |              1024 | 155.73 | 49.9 | 31132.3               |
| DS 3D | /        |          16 |  2 |  2 |  4 |             8 |              64 |              2048 | 154.84 | 49.6 | 26309.4               |
| DS 3D | /        |          32 |  2 |  2 |  8 |             8 |              64 |              4096 | 154.44 | 49.5 | 23901.8               |
| DS 3D | /        |          64 |  2 |  2 | 16 |             8 |              64 |              8192 | 153.93 | 49.3 | 22695.7               |

## Automatic Training Optimization 

If the users are not sure which strategy would achieve the largest throughput, auto_accelerate is able to automatically searching the best strategy given models and hardware conditions.  This is achieved using Bayesian Optimization (BO)implemented by [HEBO](https://github.com/huawei-noah/HEBO) aiming to find the strategy with largest training throughput efficiently. BO is a machine learning technique used for optimizing black-box functions that are expensive to evaluate. Specifically, BO learns the mapping from strategies to throughput using the data from dryrun, which runs few training steps. Moreover, BO recommends the potential high-throughput strategy to achieve the throughput from dryrun, and updates the mapping iteratively. This iterative process continues until the desired optimization criteria are met or a predefined budget is exhausted.


This fully-automatic training example is implemented by calling auto_accelerate API without load_strategy argument, so that BO algorithm would find the best optimization method combination to achieve the largest throughput.

### Fine-tuning or Pre-training
1. Fine-tuning:
Specify `--model_name_or_path`, and the script will load the `.bin` files within that directory.
```shell
python -m atorch.distributed.run --fault_tolerant --max_restarts=0 \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    bayes_opt_sg_llama2.py \
    --model_name_or_path $PRETRAINED_MODEL_DIR 
```
2. Pre-training: 
Specify `--config_name` and `--tokenizer_name`. The script will not load the `.bin` files within that directory, and instead initialize a model randomly using `config.json`.
```shell
python -m atorch.distributed.run --fault_tolerant --max_restarts=0 \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    bayes_opt_sg_llama2.py \
    --config_name $PRETRAINED_MODEL_DIR \
    --tokenizer_name $PRETRAINED_MODEL_DIR \
```

### Scripts


- training file [bayes_opt_sg_llama2.py](bayes_opt_sg_llama2.py)

- launch script [bayes_opt_sg_llama2_entry.sh](bayes_opt_sg_llama2_entry.sh)

```bash
cd dlrover/atorch/examples/llama2

# Configurable environment variable: DATASET_PATH, MODEL_NAME_OR_PATH, PER_DEVICE_TRAIN_BATCH_SIZE, etc.
# Change BO_SG_MAX_IETR(the maximum search rounds of the BO), RANDOM_SAMPLE(the initial sampling steps of BO) if needed.
bash bayes_opt_sg_llama2_entry.sh

```

### Results

- Model: Llama2-7b
- Cluster: 8 A100-80GB (NVLink) GPUs in one node.

| batch size per gpu               | Strategy |  DryRun Throughput (samples/s)  | 
|:----------------------------|:-----------------:|:-------:|
| 4  |     module_replace+ amp_native+zero2+ddp   | 11.28  |


Strategy:
```
        amp        zero             parallel_mode  module_replace  checkpoint
0   NotChosen   NotChosen  {"data": 8, "tensor": 1}       NotChosen   NotChosen
1  amp_native  zero2_fsdp  {"data": 8, "tensor": 1}  module_replace  checkpoint
2  amp_native        fsdp  {"data": 8, "tensor": 1}  module_replace  checkpoint
3  amp_native  zero2_fsdp  {"data": 8, "tensor": 1}  module_replace  checkpoint
4  amp_native        fsdp  {"data": 8, "tensor": 1}  module_replace  checkpoint
5  amp_native   NotChosen  {"data": 8, "tensor": 1}  module_replace   NotChosen
6  amp_native       zero1  {"data": 8, "tensor": 1}  module_replace  checkpoint
7  NotChosen        zero2  {"data": 8, "tensor": 1}  NotChosen         NotChosen
```

Corresponding Dryrun Throughput

```
[[ 0.        ]
 [11.28738554]
 [11.15851216]
 [11.27479499]
 [11.16217471]
 [ 0.        ]
 [ 0.        ]
 [ 0.        ]
 ]
 ```

The zero throughput in the above means that OOM occurs due to the unappropriated strategy combination.
