# Intro
- design doc

    [ds 3d parallel](design.md)

- Smaller costly transform the models that implemented by native Torch APIs (huggingface transformers, etc.) to hybrid 3D parallelism based on deepspeed pipelines and AtorchTPLayer.

# API

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
      - `custom_patcher`: `Optional[Dict[str, patch_fn]]`. `str` is the path_name of module to be patched, from the module_list that `PipeModuleFromRecordedMeta` traverses. `patch_fn` will patch the forward funciton when building the corresponding LayerSpec. For detail please refer to the GPT2 example in [design](design.md).
      - `tie_first`: whether to tie the first module in deepspeed pipeline (e.g. vocab embedding in transformer). 
      - `logit_helper`: take effect when `tie_first` is True. The helper function to compute logti on the tied embedding. `_default_logit_helper` will be used if None.
      - `ds_config`: `Dict` for deepspeed config or `str` for json file path. 
      - `batch_fn`:  input adaptation function for deepspeed pipeline (inputs,), (labels,). 
- `auto_accelerate` user interface
    <details>
    <summary>user interface</summary>


    ```python
    from transformers.xxx import XXXConfig, XXXModel

    from atorch.auto.opt_lib.ds_3d_parallel_optimization import DeepSpeed3DParallelConfig
    from atorch.utils.meta_model_utils import record_module_init

    # init distributed environment and create 3d parallel groups
    atorch.init_distributed("nccl")
    create_parallel_group(([("tensor", tensor_size), ("data", data_size), ("pipeline", pipeline_size)], None))

    # meta model for ds 3d parallel
    with record_module_init():
        meta_model = XXXModel(XXXConfig(...))

    # tensor parallel info and pipeline forward patcher
    ds_3d_parallel_cfg = DeepSpeed3DParallelConfig(
            tpinfo=get_xxx_tpinfo(),
            custom_patcher=get_xxx_custom_patcher(),
        )
    strategy = [("deepspeed_3d_parallel", ds_3d_parallel_cfg),]

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

#Example
llama2
- file entrance
[ds_3d_llama2.py](ds_3d_llama2.py)

- starup script
[llama2_entry.sh](llama2_entry.sh)

- utils
[utils.py](utils.py)