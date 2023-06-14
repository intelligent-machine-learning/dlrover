"""
Implementation of basic distributed layers. All layers support delayed initialization.
"""
import copy
import os

import psutil
import torch
import torch.distributed as dist
import torch.nn.init as init
from torch.nn.parameter import Parameter

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import (
    local_rank,
    parallel_group,
    parallel_group_and_ranks,
    parallel_group_size,
    parallel_rank,
)
from atorch.modules.distributed_modules.mappings import (
    copy_to_group,
    reduce_from_group,
    split_shard_dim_with_reshuffle_check,
)
from atorch.modules.distributed_modules.utils import VocabUtility, divide
from atorch.utils.meta_model_utils import reload_meta_module


def _memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def _initialize_affine_weight(
    weight,
    per_partition_size,
    partition_dim,
    init_method=None,
    master_weight=None,
    out_features=None,
    in_features=None,
    group_name="tensor",
    stride=1,
    return_master_weight=False,
    requires_grad=True,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = parallel_group_size(group_name)
    if world_size == 1:
        if master_weight:
            weight = master_weight
        elif init_method is not None:
            init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    if master_weight is None:
        master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=requires_grad)
        if init_method is not None:
            init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = parallel_rank(group_name)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


# FIXME This assumes that input_ is a 3D tensor,  add support for 2D tensor
# FIXME In megatron, fp16 computation here is disabled with torch.cuda.amp.autocast(enabled=False)
# FIXME Consider a fused addbmm to replace seperate matmul and bias addition
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient accumulation
    fusion in backprop.

    Adapted from Megatron-LM https://github.com/NVIDIA/Megatron-LM

    Support also the transformers gpt2 Conv1D style implementation
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group=None, async_grad_allreduce=False, weight_transpose=True):
        if process_group is None:
            process_group = "tensor"
        if isinstance(process_group, str):
            process_group = parallel_group(process_group)
        ctx.process_group = process_group
        ctx.use_bias = bias is not None
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.weight_transpose = weight_transpose
        ctx.save_for_backward(input_, weight)
        if weight_transpose:
            output = torch.matmul(input_, weight.t())
        else:
            output = torch.matmul(input_, weight)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        weight = weight.to(grad_output.dtype)
        input_ = input_.to(grad_output.dtype)
        if ctx.weight_transpose:
            grad_input = torch.matmul(grad_output, weight)
        else:
            grad_input = torch.matmul(grad_output, weight.t())

        # Convert the tensor shapes to 2D for execution compatibility
        if len(list(input_.size())) == 3:
            grad_output = grad_output.contiguous().view(
                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
            )
            input_ = input_.contiguous().view(input_.shape[0] * input_.shape[1], input_.shape[2])

        # FIXME async_op turned on
        if ctx.async_grad_allreduce:
            handle = torch.distributed.all_reduce(grad_input, group=ctx.process_group, async_op=True)

        if ctx.weight_transpose:
            grad_weight = grad_output.t().matmul(input_)
        else:
            grad_weight = torch.matmul(input_.t(), grad_output)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


class ATorchTPLayer(torch.nn.Module):
    """A base class that handles common functionalities including:
    registering process groups, ranks, original modules
    abstract reset_parameters method: used when the true initialization should be delayed
    """

    def __init__(self, orig_module=None, process_group="tensor", ranks=None, defer_init=False):
        # We must create the Parameters even we defer initialization, so that these parameters are transparent to FSDP
        # Since FSDP requires all parameters to be on the same device, we must manually clear all parameters to "meta"
        # and wait the _reset_parameters call to materialize them back.
        super().__init__()
        if process_group is None:
            process_group = "tensor"
        if isinstance(process_group, str):
            process_group, ranks = parallel_group_and_ranks(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.process_group = process_group
        self.ranks = ranks

        # A Hack to prevent PyTorch from automatically tracking orig_module
        object.__setattr__(self, "orig_module", orig_module)
        self.defer_init = defer_init

    def _reset_parameters(self):
        raise ValueError("All TP layers must implement _reset_parameters")

    def reset_parameters(self):
        tp_debug = logger.root.level > 30

        if self.orig_module is None:
            # call into _reset_parameters directly
            self._reset_parameters()
        else:
            orig_type = type(self.orig_module)
            if tp_debug and local_rank() == 0:
                mu_before_reload = _memory_usage() / (1000**3)
                logger.info(f"[TP DEBUG]: Memory before reload {orig_type}: {mu_before_reload} at rank: {local_rank()}")
            reload_meta_module(self.orig_module, delete_ckpt_name=False)

            if tp_debug and local_rank() == 0:
                mu_after_reload = _memory_usage() / (1000**3)
                logger.info(f"[TP DEBUG]: Memory after reload {orig_type}: {mu_after_reload} at rank: {local_rank()}")

            # call into _reset_parameters to shard the original module
            self._reset_parameters()

            if tp_debug and local_rank() == 0:
                mu_after_replace = _memory_usage() / (1000**3)
                logger.info(f"[TP DEBUG]: Memory after replace {orig_type}: {mu_after_replace} at rank: {local_rank()}")

            # throw away the orig_target
            # Possibly due the internals of PyTorch, del cannot release the memory taken up by orig_target
            # But .to("meta") can
            # FIXME more tests to see if this is causing trouble
            self.orig_module.to("meta")
            # remove orig_module so that the self module can pass the is_meta test
            del self.orig_module

            # clear parameters and modules
            if "orig_module" in self._modules:
                del self._modules["orig_module"]

            def delete_orig_module(module):
                for _, submodule in list(module.named_modules()):
                    if hasattr(submodule, "orig_module"):
                        delattr(submodule, "orig_module")
                        if "orig_module" in submodule._modules:
                            del submodule._modules["orig_module"]

            delete_orig_module(self)

            if tp_debug and local_rank() == 0:
                mu_after_release = _memory_usage() / (1000**3)
                logger.info(f"[TP DEBUG]: Memory after release {orig_type}: {mu_after_release} at rank: {local_rank()}")


class RowParallelLinear(ATorchTPLayer):
    """Linear layer with row parallelism.
    Adapted from Megatron-LM https://github.com/NVIDIA/Megatron-LM

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        orig_module: The original torch.nn.Linear. If this is given, we construct the
                     parallel model using exactly the parameters of the orig_module
        process_group: the process group on which to distribute this module,
                     default to parallel group 'model'
        ranks: ranks of the process group, when process_group is a str, ranks
            can be None. This must be a list of ranks (in default group), not necessarily
            sorted/contiguous
    """

    def __init__(
        self,
        input_size=None,
        output_size=None,
        bias=True,
        input_is_parallel=True,
        init_method=init.xavier_normal_,
        stride=1,
        params_dtype=torch.float32,
        orig_module=None,
        requires_grad=True,
        process_group="tensor",
        ranks=None,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if self.orig_module is not None:
            # In this case we assume that input is already properly sharded
            # Assume initialization on cuda
            input_is_parallel = True
            self.input_is_parallel = input_is_parallel
            self.input_size = orig_module.in_features
            self.output_size = orig_module.out_features
            self.params_dtype = orig_module.weight.dtype
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.use_bias = bias
            self.input_is_parallel = input_is_parallel
            self.params_dtype = params_dtype

        self.stride = stride
        self.init_method = init_method
        self.requires_grad = self.orig_module.weight.requires_grad if self.orig_module else requires_grad
        self.input_size_per_partition = divide(self.input_size, self.world_size)

        self.weight = Parameter(
            torch.empty(self.output_size, self.input_size_per_partition, dtype=self.params_dtype, device="meta")
        )

        if self.orig_module:
            if self.orig_module.bias is not None:
                my_bias = torch.empty(self.orig_module.bias.shape, dtype=self.params_dtype, device="meta")
            else:
                my_bias = None
        elif self.use_bias:
            my_bias = torch.empty(self.output_size, dtype=self.params_dtype, device="meta")
        else:
            my_bias = None

        if my_bias is not None:
            self.bias = Parameter(my_bias, self.requires_grad)
        else:
            self.register_parameter("bias", None)

        if not self.defer_init:
            self.reset_parameters()

    def _init_bias(self):
        if self.orig_module:
            my_bias = self.orig_module.bias
        elif self.use_bias:
            my_bias = torch.empty(self.output_size, dtype=self.params_dtype)
        else:
            my_bias = None

        if my_bias is not None:
            self.bias = Parameter(my_bias, self.requires_grad)
            if self.orig_module is None:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def _reset_parameters(self):
        self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, dtype=self.params_dtype))
        master_weight = self.orig_module.weight if self.orig_module else None

        # initialize weight
        _initialize_affine_weight(
            self.weight,
            per_partition_size=self.input_size_per_partition,
            partition_dim=1,
            init_method=self.init_method,
            master_weight=master_weight,
            out_features=self.output_size,
            in_features=self.input_size,
            group_name="tensor",
            stride=self.stride,
            return_master_weight=False,
            requires_grad=self.requires_grad,
        )
        self._init_bias()

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return orig_module.in_features % world_size == 0

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = split_shard_dim_with_reshuffle_check(input_, -1, self.process_group, self.ranks)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel,
            self.weight,
            None,
            self.process_group,
            False,
        )
        # All-reduce across all the partitions.
        output_ = reduce_from_group(output_parallel, self.process_group)
        return output_ + self.bias if self.bias is not None else output_


class ColumnParallelLinear(ATorchTPLayer):
    """Linear layer with column parallelism.
    Adapted from Megatron-LM https://github.com/NVIDIA/Megatron-LM

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p]

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        orig_module: The original torch.nn.Linear. If this is given, we construct the
                     parallel model using exactly the parameters of the orig_module
        process_group: the process group on which to distribute this module,
                     default to parallel group 'model'
        ranks: ranks of the process group, when process_group is a str, ranks
            can be None. This must be a list of ranks (in default group), not necessarily
            sorted/contiguous

    Returns:
        Each GPU will hold its copy Y_i = XA_i
    """

    def __init__(
        self,
        input_size=None,
        output_size=None,
        bias=True,
        init_method=init.xavier_normal_,
        stride=1,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        async_grad_allreduce=False,
        requires_grad=True,
        params_dtype=torch.float32,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if self.orig_module is not None:
            # In this case we assume that input is already properly sharded
            # Assume initialization on cuda
            input_is_parallel = True
            self.input_is_parallel = input_is_parallel
            self.input_size = self.orig_module.in_features
            self.output_size = self.orig_module.out_features
            self.params_dtype = self.orig_module.weight.dtype
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.use_bias = bias
            self.input_is_parallel = input_is_parallel
            self.params_dtype = params_dtype

        self.stride = stride
        self.init_method = init_method
        self.async_grad_allreduce = async_grad_allreduce
        self.output_size_per_partition = divide(self.output_size, self.world_size)
        self.requires_grad = self.orig_module.weight.requires_grad if self.orig_module else requires_grad

        self.weight = Parameter(
            torch.empty(self.output_size_per_partition, self.input_size, dtype=self.params_dtype, device="meta")
        )
        if self.orig_module is not None:
            if self.orig_module.bias is not None:
                my_bias = torch.empty(self.output_size_per_partition, dtype=self.params_dtype, device="meta")
            else:
                my_bias = None
        elif self.use_bias:
            my_bias = torch.empty(self.output_size_per_partition, dtype=self.params_dtype, device="meta")
        else:
            my_bias = None

        if my_bias is not None:
            self.bias = Parameter(my_bias)
        else:
            self.register_parameter("bias", None)

        if not self.defer_init:
            self.reset_parameters()

    def _init_bias(self):
        my_bias = torch.empty(self.bias.shape, dtype=self.params_dtype) if self.bias is not None else None

        if self.orig_module is not None:
            if self.orig_module.bias is not None:
                if self.orig_module.bias.is_meta:
                    master_bias = torch.load(self.orig_module.bias.checkpoint_name)
                else:
                    master_bias = copy.deepcopy(self.orig_module.bias)
            else:
                master_bias = None
        else:
            master_bias = None

        if my_bias is not None and master_bias is not None:
            _initialize_affine_weight(
                my_bias,
                per_partition_size=self.output_size_per_partition,
                partition_dim=0,
                init_method=None,
                master_weight=master_bias,
                group_name="tensor",
                stride=self.stride,
                return_master_weight=False,
                requires_grad=self.requires_grad,
            )

        if my_bias is not None:
            self.bias = Parameter(my_bias, self.requires_grad)
            if self.orig_module is None:
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def _reset_parameters(self):
        self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, dtype=self.params_dtype))
        master_weight = self.orig_module.weight if self.orig_module else None

        # initialize weight
        _initialize_affine_weight(
            self.weight,
            per_partition_size=self.output_size_per_partition,
            partition_dim=0,
            init_method=self.init_method,
            master_weight=master_weight,
            out_features=self.output_size,
            in_features=self.input_size,
            group_name="tensor",
            stride=self.stride,
            return_master_weight=False,
            requires_grad=self.requires_grad,
        )
        self._init_bias()

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return orig_module.out_features % world_size == 0

    def forward(self, input_):
        # Matrix multiply.
        if self.async_grad_allreduce:
            input_parallel = input_
        else:
            input_parallel = copy_to_group(input_, group=self.process_group)

        # Do not gather the output
        output_ = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel,
            self.weight,
            self.bias,
            self.process_group,
            self.async_grad_allreduce,
        )
        return output_


class VocabParallelEmbedding(ATorchTPLayer):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.

    Adapted from Megatron-LM https://github.com/NVIDIA/Megatron-LM
    """

    def __init__(
        self,
        num_embeddings=None,
        embedding_dim=None,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        init_method=init.xavier_normal_,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        params_dtype=torch.float32,
        requires_grad=True,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)
        self.tensor_model_parallel_size = self.world_size
        global_rank = dist.get_rank()

        self.num_embeddings = orig_module.num_embeddings if orig_module else num_embeddings
        self.embedding_dim = orig_module.embedding_dim if orig_module else embedding_dim
        self.padding_idx = orig_module.padding_idx if orig_module else padding_idx
        self.max_norm = orig_module.max_norm if orig_module else max_norm
        self.norm_type = orig_module.norm_type if orig_module else norm_type
        self.scale_grad_by_freq = orig_module.scale_grad_by_freq if orig_module else scale_grad_by_freq
        self.sparse = orig_module.sparse if orig_module else sparse
        self.requires_grad = orig_module.weight.requires_grad if orig_module else requires_grad
        self.params_dtype = orig_module.weight.dtype if orig_module else params_dtype
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, global_rank, self.ranks
        )
        self.init_method = init_method
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                requires_grad=self.requires_grad,
                dtype=self.params_dtype,
                device="meta",
            )
        )

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        # Allocate weights.
        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                requires_grad=self.requires_grad,
                dtype=self.params_dtype,
            )
        )
        master_weight = self.orig_module.weight if self.orig_module else None
        _initialize_affine_weight(
            self.weight,
            per_partition_size=self.num_embeddings_per_partition,
            partition_dim=0,
            init_method=self.init_method,
            master_weight=master_weight,
            out_features=self.num_embeddings,
            in_features=self.embedding_dim,
            group_name="tensor",
            requires_grad=self.requires_grad,
        )

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return orig_module.num_embeddings % world_size == 0

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = torch.nn.functional.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        # FIXME use a backward enabled operator
        output = reduce_from_group(output_parallel, self.process_group)
        return output
