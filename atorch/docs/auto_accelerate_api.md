# auto_accelerate API

ATorch supports efficient and easy-to-use model training experience by decoupling model definition from training optimization strategy. The decoupling ability is provided by auto_accelerate API. 
The auto_accelerate API provides two ways of usage:

1. Fully automatic mode: Automatically generates optimization strategy and implements automatic optimization.
2. Semi-automatic mode: Users specify the method used for optimization strategy through load_strategy, and auto_accelerate automatically configures and implements the optimization strategy.

An optimization strategy consists of multiple optimization methods, including parallel training, GPU memory optimization methods, compute optimization, etc. Supported optimization methods are listed in [doc link](#supported-optimization-methods). 

## Inputs
auto_accelerate takes model, optim_func, dataset, loss_func, etc as inputs, and generates optimized model, optimizer, dataloader etc as outputs.
```python
auto_accelerate(
    model,
    optim_func=None,
    dataset=None,
    loss_func=None,
    prepare_input=None,
    model_input_format=None,
    optim_args=None,
    optim_param_func=None,
    dataloader_args=None,
    distributed_sampler_cls=None,
    excluded=None,
    included=None,
    load_strategy=None,
    lr_scheduler_cls=None,
    lr_scheduler_args=None,
    finetune_strategy=False,
    save_strategy_to_file=None,
    **kargs,
)
```

<table>
  <tr>
    <th>API Parameter</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>model (torch.nn.Module) </td>
    <td> A pytorch model (non-distributed version) </td>
  </tr>
   <tr>
    <td>optim_func (Callable) </td>
    <td>Either a pytorch optimizer function such as torch.optim.ADAMW，or a user-defined function takes input as (parameters, **optim_args). Example:  
    <code>

    def optim_func(parameters, **optim_args):
        return optim.SGD(parameters, **optim_args)
</code></td>
  </tr>
  
  <tr>
    <td>dataset (torch.utils.data.Dataset)</td>
    <td>dataset for training</td>
  </tr>

  <tr>
    <td>loss_func(Callable)</td>
    <td>loss function takes model input and model output as its parameters, return loss (scaler) or a list/tuple with its first item as loss. Example: <code>

    def loss_func(input, output):
        loss = nn.MSELoss()
        return loss(input["label"], output) 
</code>
</td>
</tr>

<tr>
    <td>prepare_input (Callable, optional)</td>
    <td>If None, data from dataloader would move to corresponding device.If provided, function should take (data, device) as input, and output as model input. Example: <code>
    
    def prepare_input(data, device):
        return transform(data["sample"]).to(device), data["label"].to(device)
</code></td>
  </tr>

  <tr>
    <td>optim_args (Dict, optional)</td>
    <td>optimizer arguments, such as：<code>
    
    optim_args = {
     "lr": 0.01, 
     "momentum": 0.9
    }
</code></td>
  </tr>

<tr>
    <td>optim_param_func (Callable, optional) </td>
    <td>Supports per-parameter options for optimizer. Example: <code>

    def optim_param_func(model):
        parameters = [{'params': model.base.parameters()},
            {'params': model.classifier.parameters(), 'lr': 1e-3}]
        return parameters
</code></td>
  </tr>

  <tr>
    <td>dataloader_args (Dict, optional) </td>
    <td>dataloader arguments, such as batch_size、shuffle、sampler, etc.
Note that strong scaling is used, thus batch_size is the global batch_size in training, not per-process batch_size.</td>
  </tr>

  <tr>
    <td>distributed_sampler_cls (type, optional)</td>
    <td>used-defined distributed sampler with same interface as torch.utils.data.distributed.DistributedSampler. </td>
  </tr>

  <tr>
    <td>model_input_format (str, optional)</td>
    <td>dataloader data format.

None: use data as model input.

"unpack_sequence": use *data as model input.

"unpack_dict": use **data as model input.</td>
  </tr>

  <tr>
    <td>excluded (List[str], optional) </td>
    <td>optimization methods excluded in fully-automatic mode.</td>
  </tr>

  <tr>
    <td>included (List[str], optional)</td>
    <td> (Not supported yet) optimization methods must be included in fully-automatic mode.</td>
  </tr>

<tr>
    <td>load_strategy (optional)</td>
    <td>If None, fully-automatic model.

Not None, semi-automatic model. Supported formats:

1. str: strategy(str)

2. bytes: strategy(bytes)

3. `atorch.auto.strategy.Strategy` instance.

4. list consists of optimization_name and/or (optimization_name, optimization_config).</td>
  </tr>


  <tr>
    <td>ignore_dryrun_on_load_strategy (optional, default True)</td>
    <td>
       If True, ignore dryrun when load_strategy is not None.
    </td>
  </tr>

  <tr>
    <td>finetune_strategy (optional, default False)</td>
    <td>
       If True and load_strategy is not None, finetune loaded strategy. 
    </td>
  </tr>

  <tr>
    <td>save_strategy_to_file (str, optional)</td>
    <td>
        If not None, save strategy to file using pickle.dumps
    </td>
  </tr>

  <tr>
    <td>kargs (Dict, optional)</td>
    <td>  Other optional arguments. Supported args:

    "verbose"：(bool) default False. If True, print more info.

    "time_limit"：time limit (second) for fully-automatic mode.

    "find_unused_parameters"：(bool): default False. parameters for DDP (when DDP is in optimization methods).

</td>
  </tr>
</table>

## Outputs

auto_accelerate returns 3-item tuple (status, result, best_strategy).

<table>
  <tr>
    <th>Return Item</th>
    <th>Description</th>
  </tr>

<tr>
    <td>status (bool)</td>
    <td>
        If auto_accelerate succeeds.
    </td>
</tr>

<tr>
    <td>result (namedtuple)</td>
    <td>
    namedtuple(model, optim, dataloader, loss_func, prepare_input, args) for resulting model, optimizer, dataloader, loss_func, prepare_input, args. <br>
    args includes:<br>
    - use_optim_backward: if True, use <code>optim.backward(loss)</code> instead of <code>loss.bacward()</code> for backward pass.<br>
    - requires_set_gradient_accumulation_boundary: if True, when gradient accumulation is used in traning, call <code>optim.set_gradient_accumulation_boundary(True)</code> in accumulation boundary training pass.    
    </td>
</tr>
<tr>
    <td>best_strategy (atorch.auto.strategy.Strategy)</td>
    <td>
        The resulting optimization strategy when status is True.
    </td>
</tr>
  </table>

## Supported Optimization Methods

During semi-automatic mode auto_accelerate optimization, load_strategy is used to specify the optimization strategy. An optimization strategy is a combination of multiple optimization methods (a list of optimization methods).  Some optimization methods support configuration options. Therefore, load_strategy is a list, where each list item is either an optimization method name (opt_name) or a tuple consisting of an optimization method name and a config (opt_name, config). 

If an optimization method requires a config, it can be specified during semi-automatic optimization, or the default value can be used (not specified).  
For example, load_strategy = ["paralle_mode", ("amp_native", "bf16")] would use DDP parallel with automatic mixed precision with bf16.
When specifying the optimization strategy using load_strategy, ignore_dryrun_on_load_strategy=True can be used to skip the dryrun step and accelerate the auto_accelerate process.

When auto_accelerate succeeds, the returned best_strategy or saved strategy (saved in save_strategy_to_file) can be reused as load_strategy to speedup auto_accelerate.

Below is a list of supported optimization methods.


### parallel_mode

parallel_mode is a special optimization method used to specify (1)
-  the use of distributed training; 
- the grouping of different parallel methods if it is a hybrid parallel mode. 

The default configuration is data parallelism, where all processes are in one process group and perform data parallelism.

The configuration format is: 

<code>(List[Tuple[str, int]], Optional(List(int)))</code>

The first item is a list of <code>(name, size)</code>, specifying the groups for hybrid parallelism. The product of all sizes is the number of processes (world_size) in distributed training.
The second item is the rank order. If it is None, it means sequential rank order.
For example, <code>([("tensor", 4), ("pipeline", 2), ("data", 2)], None)</code> represents the use of 3D parallelism, with <code>[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]</code> 4 groups for tensor parallel, 
 for "pipeline" <code>[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]</code>  8 process groups for pipeline parallel, and 
<code>[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]</code> 8 process groups for data parallel.

The supported types for <code>name</code> are: <code>data, zero, tensor, pipeline</code>, corresponding to data parallelism, zero data parallelism, tensor parallelism, and pipeline parallelism, respectively.
The size of <code>data</code> will be automatically adjusted so that the product of all sizes equals world_size. For example, in training with 16 cards (world_size=16), if no config is specified, the default is <code>([("data", 16)], None)</code>; If config is specified as <code>([("data", 8)], None)</code>, it will be automatically adjusted to <code>([("data", 16)], None)</code>; If set config as <code>([("tensor", 2), ("pipeline", 2), ("data", 2)], None)</code>, it will be automatically changed to <code>([("tensor", 2), ("pipeline", 2), ("data", 4)], None)</code>.

### amp_native

amp_native is an optimization method for automatic mixed precision, which uses the PyTorch-native amp implementation.
The default configuration is <code>{"dtype": torch.float16}</code>, which uses fp16 for mixed precision.
It automatically scales and checks gradient values. If there is any exception (inf/nan), it skips that step.
If you want to use bf16 mixed precision, can set config as <code>{"dtype": torch.bfloat16}</code>.
For bfloat16, it does not check if the gradients are infinite by default. If you want to check and skip the step, you can add <code>skip_if_nonfinite</code> to the config, such as  <code>{"dtype": torch.bfloat16, "skip_if_nonfinite": True}</code>.


### half

Training in half precision. Default configuration is <code>"fp16"</code>. If want to use bfloat16, set config as <code>"bf16"</code>.

### fp8

Use the FP8 capability provided by [transformer_engine](https://github.com/NVIDIA/TransformerEngine) (te) to accelerate computation. This optimization method will automatically replace <code>nn.Linear</code> module in the model with <code>te.Linear</code> to speed up computation. fp8 is compatible with other optimization methods such as [amp_native](#amp_native), [half](#half), [fsdp](#fsdp), [checkpoint](#checkpoint), etc. 
Note that lora([peft](https://github.com/huggingface/peft)) fp8 training is not supported yet.

**Pre-requisites**
- Hardware support: GPU sm >=8.9 (such as Ada, Hopper, etc.). If not satisfied, fp8 optimization will be ignored.
- Software support: transformer_engine installed, version >= 1.0.
- Tensor dimension requirements: For tensor core fp8 computation, tensor dim[0] must be a multiple of 8, and dim[1] must be a multiple of 16. Since the backward computation of <code>nn.Linear</code> during training requires a transpose op, this means that both the weight of <code>nn.Linear</code> and the module's input need dim[0] and dim[1] to be multiples of 16. For weight dimensions, fp8 optimization method will check automatically, and it is up to the users to ensure that the input to <code>nn.Linear</code> also meets this dimension requirement.

**Supported config parameters**

```
include: List[str], default None.
    If None, all nn.Linear module can use te.
    If not None, nn.Linear module name should have at least one substring equals to items in include.
exclude: List[str], default None.
    If None, all modules that passing include test would use te.
    If not None, if a nn.Linear module name has at least one substring matches exclude, it will not use te.
verbose: Bool, default False. If True, print names of those submodules that are replaced by  <code>te.Linear </code>.
recipe.DelayedScaling parameter:
    margin: default 0
    interval: default 1
    fp8_format: “HYBRID” (default) or “E4M3”
    amax_history_len: default 1024
    amax_compute_algo: “max” (default) or “most_recent”
    reduce_amax: default True
```

**Default config**
```
{"include": None, "exclude": None, "margin": 0, "interval": 1, "fp8_format": "HYBRID", "amax_history_len": 1024, "amax_compute_algo": "max", "reduce_amax": True}
```

All <code>nn.Linear</code> instances that pass the "include" and "exclude" conditions and whose weight dim[0] and dim[1] are multiples of 16 will be automatically converted to <code>te.Linear</code>, using <code>recipe.DelayedScaling</code> defined by the config parameters excluding "include" and "exclude" for automatic fp8 computation.

**Example**

In a [llama](https://github.com/huggingface/transformers/blob/53cffeb33c2f46ccef8969da076aa620f70801c4/src/transformers/models/llama/modeling_llama.py#L1106) model, <code>nn.Linear</code> exists not only in the <code>LlamaDecoderLayer</code> but also <code>lm_head</code> . Using fp8 training for <code>nn.Linear</code> in  <code>LlamaDecoderLayer</code> usually does not affect the convergence accuracy, but it has a severe impact when <code>lm_head</code> also uses fp8. In this case, you can use the config so that the module replacement only affects <code>the LlamaDecoderLayer</code> and not the <code>lm_head</code>.

This can be achieved using <code>include</code> config parameter:

<code>config = {"include": ("layers",)}</code>

Or using <code>exclude</code> config parameter:

<code>config = {"exclude": ("lm_head",)}</code>



### module_replace

Automatic module optimization, which replaces optimizable modules with optimized implementations.
Supported optimized modules are:

- FlashAttention, already adapted for BertAttention, GPT2Attention, CLIPAttention, LlamaAttention.
- FusedLayerNorm, replaces LayerNorm.

Flash attention is effective when used with mixed precision or half precision (i.e., in conjunction with amp_native, half). It currently supports GPU with sm version >= 8.0, such as A100, A10, H100, etc.

User-defined module optimization is supported by registering the optimized module in advance:

```
from atorch.auto.opt_lib.module_replace_optimization import register_replace_pair

supported_dtypes={torch.float32, torch.float16, torch.bfloat16}
pair_cls = (module_to_be_optimized, optimized_module)

register_replace_pair("my_optimized_module", supported_dtypes=supported_dtypes, pair_cls=pair_cls)

```

### zero1

zero1 uses Zero Redundancy Optimizer (zero1) to shard optimizer states in data parallel training.
Two implementations are provided.
- (Default) Use fairscale zero2 implementation.
- Use DeepSpeed zero1 implementation.  Set config as <code>{"use_ds_zero", True}</code> to choose this implementation.

### zero2

Level 2 of ZeRO method, which shards both gradients and optimizer states.

Three implementations are provided.
- (Default) Use pytorch fsdp SHARD_GRAD_OP, thus supports all configurations as in fsdp method below.
- Use fairscale zero2 implementation. Set config as <code>{"not_use_fsdp", True}</code> to choose this implementation.
- Use DeepSpeed zero2 implementation.  Set config as <code>{"use_ds_zero", True}</code> to choose this implementation.


### fsdp

Use PyTorch-native FSDP implementation for level 3 of ZeRO, which shards model parameters, gradients and optimizer states.
Configuration support all [FSDP arguments](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel) plus some ATorch-defined arguments for easy usage.
ATorch-defined arguments:

- atorch_wrap_cls: tuple of submodule names or submodule type for fsdp to wrap.
- atorch_size_based_min_num_params： wrap submoudule based on parameter size. Should not used with atorch_wrap_cls.
- atorch_ignored_cls: tuple of submodule names or submodule type for fsdp to ignore (not sharded).
- wrap_trainable_outmost: if True, wrap trainable parameters together in an outmost fsdp wrap. You may get better performance for finetuning with a small percentage of trainable parameters, such as LORA.

Recommended configurations
```
config = {"forward_prefetch": True, "limit_all_gathers": True, "sync_module_states": True, atorch_wrap_cls=tuple_of_main_submodules}
```

Add <code>{"use_orig_params": True}</code> if multiple parameter groups with different hyperparamters are used in optimizer.  Try add <code>{"fsdp_wrap_params_outmost": True}</code> for LORA finetuning to see if any performance improvement.

### checkpoint

Activation checkpoint is a memory-saving method which trade computation for memory. It does not keep activations during forward pass, but uses recomputation in backward pass to generate activations for gradient computation. Configuration is required to indicate which modules would be checkpointed.

Configuration can be a tuple of module types or module names, such as:
```
config = (GPT2Attention, GPT2MLP)
```

There are two checkpoint implementations in PyTorch, no_reentrant and reentrant. no_reentrant is default and its performance is better than reentrant. In some cases such that model definition contains <code>@torch.jit.script</code>, no_reentrant implementation may fail and reentrant should be used. Checkpoint configuration supports dict format to support choosing reentrant implementation.
```
config = {
  "wrap_class": (GPT2Attention, GPT2MLP), # modules to checkpoint
  "no_reentrant": False,                  # use reentrant implementation
}
```



### tensor_parallel

Tensor parallel, which would split modules in Megatron style tensor parallel automatically. The degree of tensor parallelism is specified in parallel_mode configuration, such as <code>("tensor", 8)</code> for degree = 8.

### pipeline_parallel

Pipeline parallel, which would split model in multiple stages automatically. The degree of pipeline parallelism is specified in parallel_mode configuration, such as <code>("pipeline", 4)</code> for degree = 4.


### mixed_parallel

Automatically training model with tensor parallel, pipeline parallel, and daa parallel. parallel_mode configuration would specify the degree of each parallelism. For example, <code>([("tensor", 8), ("pipeline", 2), ("data", 2)]</code> specifies 8, 2, 2 for degrees of tensor parallel, pipeline parallel, and data parallel respectively.

### ds_3d_parallel

Use DeepSpeed pipeline engine for 3D parallel.

## Examples

See [auto_accelerate examples](../examples/auto_accelerate/README.md) for detail. Moreover, please refer to [example](../examples/llama2/README.md) for fully automatic mode of auto_accelerate.

