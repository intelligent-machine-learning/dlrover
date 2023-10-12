# auto_accelerate API

ATorch supports efficient and easy-to-use model training experience by decoupling model definition from training optimization strategy. The decoupling ability is provided by auto_accelerate API. 
The auto_accelerate API provides two ways of usage:

1. Fully automatic mode: Automatically generates optimization strategy and implements automatic optimization.
2. Semi-automatic mode: Users specify the method used for optimization strategy through load_strategy, and auto_accelerate automatically configures and implements the optimization strategy.

An optimization strategy consists of multiple optimization methods, including parallel training, GPU memory optimization methods, compute optimization, etc. Supported optimization methods are listed in [doc link]. 

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
    <th><code>Description</code></th>
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
    <td>used-defined distributed sampler with same interafce as torch.utils.data.distributed.DistributedSampler. </td>
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

Not None, semi-automaticl model. Supported formats:

1. str: strategy(str)

2. bytes: strategy(bytes)

3. `atorch.auto.strategy.Strategy` instance.

4. list consists of optimization_name and/or (optimization_name, optimization_config).</td>
  </tr>


  <tr>
    <td>ignore_dryrun_on_load_strategy (optional, default False)</td>
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
    <th><code>Description</code></th>
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
        namedtuple(model, optim, dataloader, loss_func, prepare_input) for resulting model, optimizer, dataloader, loss_func, prepare_input.
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

Before is a list of supported optimization methods.




## Examples

See [auto_accelerate](../examples/auto_accelerate/README.md) examples for detail.

