import os
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from distutils.util import strtobool
from torch import Tensor
from torch.nn import Module

import atorch
from atorch.common.constants import AnalyserConstants, GPUCapability
from atorch.common.file import file_io  # noqa: F401
from atorch.common.log_utils import default_logger as logger
from atorch.utils.metrics_reporter import _EASYDL_COLLECTOR, _REPORTER_ENABLED, _UNIQUE_JOB_ID

try:
    from aistudio_common.utils import algo_metrics_utils
except Exception:
    algo_metrics_utils = None

try:
    import flash_attn.flash_attn_interface as fa

    has_flash_attn = True
except (ModuleNotFoundError, ImportError):
    has_flash_attn = False

registered_modules: Set[Module] = set()
module_flop_count = []
module_mac_count = []
old_functions = {}


pre = 0
post = 0


class AProfiler(object):
    def __init__(self, model):
        self.model = model
        self.started = False
        self.patched = False
        self.timeline_handler = None
        self.timeline_enabled = False
        self.export_report = False
        if not torch.cuda.is_available():
            raise SystemError("Cannot use AProfiler without CUDA.")

    def start_profile(self, ignore_list=None, enable_timeline=False, distributed=False, timeline_only=False, **kwargs):
        """Starts profiling.
        Extra attributes are added recursively to all the modules and the profiled
        torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
            enable_timeline(bool): enable pytorch profile timeline
            distributed(bool): if each pod export a timeline/report file
            timeline_only(bool): only export timeline, skip other analysis.
            kwargs: any params for torch.profiler.profile
        """
        self._reset_profiler()

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            if module in registered_modules:
                return

            registered_modules.add(module)

            def pre_forward_hook(module, input):

                module_flop_count.append([])
                module_mac_count.append([])
                module.__start_event = torch.cuda.Event(enable_timing=True)
                module.__end_event = torch.cuda.Event(enable_timing=True)
                module.__start_event.record()
                module.__memory_before__ = torch.cuda.memory_allocated()

            module.__pre_forward_hook_handle__ = module.register_forward_pre_hook(pre_forward_hook)

            def post_forward_hook(module, input, output):
                if module_flop_count:
                    module.__macs__ += sum([elem[1] for elem in module_mac_count[-1]])
                    module_mac_count.pop()
                    module.__flops__ += sum([elem[1] for elem in module_flop_count[-1]])
                    module_flop_count.pop()
                module.__end_event.record()
                torch.cuda.synchronize()
                module.__duration__ = module.__start_event.elapsed_time(module.__end_event) / 1000
                module.__activation__ = torch.cuda.memory_allocated() - module.__memory_before__

            module.__post_forward_hook_handle__ = module.register_forward_hook(post_forward_hook)

        if not timeline_only:
            _patch_functionals()
            _patch_tensor_methods()
            self.model.apply(partial(register_module_hooks, ignore_list=ignore_list))
            self.patched = True

        local_rank = atorch.distributed.local_rank()
        edl_enabled = strtobool(os.getenv("ELASTICDL_ENABLED", "false"))
        is_node_0 = os.getenv("POD_NAME", "").endswith("worker-0" if edl_enabled else "master-0") or atorch.rank() in (
            0,
            None,
        )
        if not distributed and not is_node_0:
            logger.warning("Not distribute mode and this pod is not node_0, skip exporting profiling file")
            return
        if local_rank == 0:
            if enable_timeline:
                self.timeline_handler = torch.profiler.profile(**kwargs)
                self.timeline_handler.start()
                self.timeline_enabled = True
            if not timeline_only:
                self.export_report = True
        self.started = True

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """

        def remove_profile_hooks(module):
            if hasattr(module, "__pre_forward_hook_handle__"):
                module.__pre_forward_hook_handle__.remove()
                del module.__pre_forward_hook_handle__
            if hasattr(module, "__post_forward_hook_handle__"):
                module.__post_forward_hook_handle__.remove()
                del module.__post_forward_hook_handle__

        if self.started and self.patched:
            _reload_functionals()
            _reload_tensor_methods()
            self.model.apply(remove_profile_hooks)
            self.patched = False

        if self.timeline_handler and self.timeline_enabled:
            self.timeline_handler.stop()
            self.timeline_enabled = False

    def end_profile(self, total_steps_time=0):
        """
        Args:
            total_steps_time: duration time of steps between start_profile() and end_profile
        """
        if not self.started:
            return
        self.stop_profile()

        def remove_profile_attrs(module):
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__macs__"):
                del module.__macs__
            if hasattr(module, "__params__"):
                del module.__params__
            if hasattr(module, "__start_time__"):
                del module.__start_time__
            if hasattr(module, "__duration__"):
                del module.__duration__
            if hasattr(module, "__memory_before__"):
                del module.__memory_before__
            if hasattr(module, "__activation__"):
                del module.__activation__

        os.makedirs(AnalyserConstants.PROFILE_DIR, exist_ok=True)
        if self.timeline_handler:
            try:
                timeline_file_path = os.path.join(AnalyserConstants.PROFILE_DIR, AnalyserConstants.TIMELINE_FILE_NAME)
                self.timeline_handler.export_chrome_trace(timeline_file_path)
                logger.info(f"Timeline profiling saved to {timeline_file_path}")
                self.timeline_handler = None
                timeline_signal_file_path = os.path.join(
                    AnalyserConstants.PROFILE_DIR, AnalyserConstants.TIMELINE_SIGNAL_FILE_NAME
                )
                with open(timeline_signal_file_path, "w") as f:
                    f.write(timeline_file_path)
            except Exception as e:
                logger.warning(f"Write timeline file to {AnalyserConstants.PROFILE_DIR} failed ! {str(e)}")
                self.timeline_handler = None
        # export aprof.txt
        if self.export_report:
            self.compute_gpu_utilization(total_steps_time)
            try:
                prof_file_path = os.path.join(AnalyserConstants.PROFILE_DIR, AnalyserConstants.PROF_FILE_NAME)
                self.print_model_profile(output_file=prof_file_path)
                logger.info(f"Profile report saved to {prof_file_path}.")
                prof_signal_file_path = os.path.join(
                    AnalyserConstants.PROFILE_DIR, AnalyserConstants.PROF_SIGNAL_FILE_NAME
                )
                with open(prof_signal_file_path, "w") as f:
                    f.write(prof_file_path)
            except Exception as e:
                logger.warning(f"Write aprof.txt file to {AnalyserConstants.PROFILE_DIR} failed ! {str(e)}")
        self.model.apply(remove_profile_attrs)
        self.started = False

    def _reset_profiler(self):
        """
        Resets the profiling.

        Adds or resets the extra attributes.
        """

        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__macs__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters() if p.requires_grad)
            module.__start_time__ = 0
            module.__duration__ = 0
            module.__memory_before__ = 0
            module.__activation__ = 0

        self.model.apply(add_or_reset_attrs)
        self.timeline_handler = None

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (boo, optional): whether to output the
            flops as string. Defaults to False.

        """
        total_flops = _get_module_flops(self.model)
        return _flops_to_string(total_flops) if as_string else total_flops

    def get_total_macs(self, as_string=False):
        """Returns the total MACs of model.

        Args:
            as_string (bool, optional): whether to  output the total macs
            as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_macs = _get_module_macs(self.model)
        return _macs_to_string(total_macs) if as_string else total_macs

    def get_total_duration(self, as_string=False):
        """Returns the total duration of the model forward pass.

        Args:
            as_string (bool, optional): whether to ouput the duration as string. Defaults to false.

        """
        total_duration = _get_module_duration(self.model)
        return _duration_to_string(total_duration) if as_string else total_duration

    def get_total_params(self, as_string=False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to false.

        Returns:
             The number of parameters in the model.
        """
        total_params = _get_module_params(self.model)
        return _params_to_string(total_params) if as_string else total_params

    def get_total_activation(self, as_string=False):
        """Returns the total activation bytes of the model.

        Args:
            as_string (bool, optional): whether to output the activation bytes as string. Defaults to false.

        Returns:
             The bytes of activation in the model.
        """
        activation = _get_module_activation(self.model)
        return _activation_to_string(activation) if as_string else activation

    def print_model_profile(
        self,
        profile_step=1,
        module_depth=-1,
        top_modules=1,
        detailed=False,
        output_file=None,
    ):
        if not self.started:
            return

        import os.path
        import sys

        original_stdout = None
        f = None
        if output_file and output_file != "":
            dir_path = os.path.dirname(output_file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            original_stdout = sys.stdout
            f = open(output_file, "w")
            sys.stdout = f

        total_flops = self.get_total_flops()
        total_macs = self.get_total_macs()
        total_duration = self.get_total_duration()
        total_params = self.get_total_params()
        total_activation = self.get_total_activation()

        print(f"\n-------------------------- AProfiler Summary at step {profile_step}--------------------------")
        print(
            "Notations:\n"
            "1. number of parameters (params),\n"
            "2. activation bytes (activation),\n"
            "3. number of multiply-accumulate operations(macs),\n"
            "4. number of floating-point operations (flops),\n"
            "5. floating-point operations per second (FLOPS),\n"
            "6. fwd latency (forward propagation latency)\n"
        )
        print("{:<60}  {:<8}".format("params per gpu: ", _params_to_string(total_params)))
        print("{:<60}  {:<8}".format("activation bytes: ", _params_to_string(total_activation)))
        print("{:<60}  {:<8}".format("fwd macs per GPU: ", _macs_to_string(total_macs)))
        print("{:<60}  {:<8}".format("fwd flops per GPU: ", _num_to_string(total_flops)))
        print("{:<60}  {:<8}".format("fwd latency: ", _duration_to_string(total_duration)))
        print(
            "{:<60}  {:<8}".format(
                "fwd FLOPS per GPU = fwd flops per GPU / fwd latency: ",
                _flops_to_string(total_flops / total_duration),
            )
        )

        def _extra_repr(module):
            params = _get_module_params(module)
            flops = _get_module_flops(module)
            macs = _get_module_macs(module)
            duration = _get_module_duration(module)
            activation = _get_module_activation(module)
            items = [
                "params: {} and {:.2%}".format(_params_to_string(params), params / total_params),
                "flops: {} and {:.2%}".format(_flops_to_string(flops), flops / total_flops),
                "macs: {} and {:.2%}".format(
                    _macs_to_string(macs),
                    0.0 if total_macs == 0 else macs / total_macs,
                ),
                "fwd latency: {} and {:.2%}".format(
                    _duration_to_string(duration),
                    0.0 if total_duration == 0 else duration / total_duration,
                ),
                "activation: {} and {:.2%}".format(
                    _activation_to_string(activation),
                    0.0 if total_activation == 0 else activation / total_activation,
                ),
                module.original_extra_repr(),
            ]
            return ", ".join(items)

        def _add_extra_repr(module):
            extra_repr = _extra_repr.__get__(module)
            if module.extra_repr != extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = extra_repr
                assert module.extra_repr != module.original_extra_repr

        def _del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr

        self.model.apply(_add_extra_repr)

        print("\n----------------------------- Aggregated Profile per GPU -----------------------------")

        self.print_model_aggregated_profile(module_depth=module_depth, top_modules=top_modules)

        if detailed:
            print("\n------------------------------ Detailed Profile per GPU ------------------------------")
            print(self.model)

        self.model.apply(_del_extra_repr)

        print("------------------------------------------------------------------------------")

        if output_file:
            sys.stdout = original_stdout
            f.close()

    def print_model_aggregated_profile(self, module_depth, top_modules=1):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops,
            and parameters ad depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional):
        """

        info = {}
        if not hasattr(self.model, "__flops__"):
            print("no __flops__ attribute in the model, call this function agter start_profile and before end_profile")
            return []

        def dfs_walk(module, cur_depth, info):
            if cur_depth not in info:
                info[cur_depth] = {}
            if module.__class__.__name__ not in info[cur_depth]:
                info[cur_depth][module.__class__.__name__] = [
                    0,
                    0,
                    0,
                    0,
                    0,
                ]  # params, flops, macs, duration, activation.
            info[cur_depth][module.__class__.__name__][0] += _get_module_params(module)
            info[cur_depth][module.__class__.__name__][1] += _get_module_flops(module)
            info[cur_depth][module.__class__.__name__][2] += _get_module_macs(module)
            info[cur_depth][module.__class__.__name__][3] += _get_module_duration(module)
            info[cur_depth][module.__class__.__name__][4] += _get_module_activation(module)
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    dfs_walk(child, cur_depth + 1, info)

        dfs_walk(self.model, 0, info)

        depth = module_depth
        if module_depth == -1:
            depth = len(info)

        print(
            f"Top {top_modules} modules in terms of params, flops, macs, fwd latency,"
            f"actication bytes at different model depths:"
        )

        for d in range(depth):
            num_items = min(top_modules, len(info[d]))
            sort_params = {
                k: _params_to_string(v[0])
                for k, v in sorted(info[d].items(), key=lambda item: item[1][0], reverse=True)[:num_items]
            }
            sort_flops = {
                k: _flops_to_string(v[1])
                for k, v in sorted(info[d].items(), key=lambda item: item[1][1], reverse=True)[:num_items]
            }

            sort_macs = {
                k: _macs_to_string(v[2])
                for k, v in sorted(info[d].items(), key=lambda item: item[1][2], reverse=True)[:num_items]
            }

            sort_durations = {
                k: _duration_to_string(v[3])
                for k, v in sorted(info[d].items(), key=lambda item: item[1][3], reverse=True)[:num_items]
            }

            sort_activation = {
                k: _activation_to_string(v[4])
                for k, v in sorted(info[d].items(), key=lambda item: item[1][4], reverse=True)[:num_items]
            }

            print(f"depth {d}:")
            print(f"    params:     - {sort_params}")
            print(f"    flops:      - {sort_flops}")
            print(f"    macs:       - {sort_macs}")
            print(f"    fwd latency - {sort_durations}")
            print(f"    activation  - {sort_activation}")

    def compute_gpu_utilization(self, total_steps_time=0, dtype="FP16"):
        if not total_steps_time:
            logger.warning("You must give total_steps_time in end_profiler() to compute gpu utilization.")
            return 0.0
        if dtype not in GPUCapability.TFLOPS.keys():
            logger.warning(f"Your dtype {dtype} is not supported.")
            return 0.0
        flops_t = self.get_total_flops() * 3 / total_steps_time / 10.0**12
        gpu_name = get_gpu_name()
        try:
            gpu_capability = GPUCapability.TFLOPS[dtype][gpu_name]
        except Exception:
            logger.warning(f"Your GPU {gpu_name} is not supported now.")
            return 0.0
        gpu_utilization = round(flops_t / gpu_capability, 2)
        _report_metrics_internal({AnalyserConstants.GPU_UTILIZATION: gpu_utilization})
        return gpu_utilization


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = torch.numel(input) * out_features
    return 2 * macs, macs


def _relu_flops_compute(input, inplace=False):
    return torch.numel(input), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return torch.numel(input), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return torch.numel(input), 0


def _leaky_relu_flops_compute(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    return torch.numel(input), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return torch.numel(input), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return torch.numel(input), 0


def _gelu_flops_compute(input, *args, **kwargs):
    return torch.numel(input), 0


def _pool_flops_compute(input, *args, **kwargs):
    return torch.numel(input), 0


def _conv_flops_compute(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[-2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return torch.numel(input) * (5 if has_affine else 4), 0
    flops = torch.numel(input) * (2 if has_affine else 1)
    return flops, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _group_norm_flops_compute(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _upsample_flops_compute(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if size is not None:
        if isinstance(size, (tuple, list)):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    assert scale_factor is not None, "either size or scale_factor should be defined"
    flops = torch.numel(input)
    if isinstance(scale_factor, (tuple, list)) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor ** len(input)
    return flops, 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    x = input[0]
    nfeatures = x.size()[dim] if dim else x.size()
    batch_size = x.numel() // nfeatures
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = int(batch_size * (total_exp + total_add + total_div))
    flops = batch_size * total_ops
    return flops, batch_size * total_add


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0, 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")

    def get_input_shape(operands):
        for o in operands:
            if isinstance(o, (tuple, list)):
                yield from get_input_shape(o)
            elif isinstance(o, Tensor):
                yield o.shape

    input_shapes = list(get_input_shape(operands))

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _mul_flops_compute(input, other, *, out=None):
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input, other, *, alpha=1, out=None):
    return _elementwise_flops_compute(input, other)


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0


def _flash_attn_flops_compute(
    qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale=None, causal=False, return_attn_probs=False
):
    """qkv: (total, 3, nheads, headdim), cu_seqlens: (batch_size + 1,), max_seqlen: int"""
    _, _, nheads, headdim = qkv.shape
    batch_size = cu_seqlens.shape[0] - 1
    qk_macs = batch_size * nheads * (max_seqlen**2) * headdim
    fake_tensor = torch.zeros([batch_size, nheads, max_seqlen, max_seqlen])
    softmax_flops, softmax_macs = _softmax_flops_compute(fake_tensor)

    multi_v_macs = batch_size * nheads * (max_seqlen**2) * headdim
    total_flops = 2 * qk_macs + max_seqlen**2 + softmax_flops + 2 * multi_v_macs
    total_macs = qk_macs + softmax_macs + multi_v_macs
    return total_flops, total_macs


def _wrap_func(func, flop_compute):
    oldFunc = func
    name = func.__name__
    old_functions[name] = oldFunc

    def newFunc(*args, **kwargs):
        flops, macs = flop_compute(*args, **kwargs)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        return oldFunc(*args, **kwargs)

    newFunc.__name__ = func.__name__
    return newFunc


def _patch_functionals():
    # FC
    F.linear = _wrap_func(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = _wrap_func(F.conv1d, _conv_flops_compute)
    F.conv2d = _wrap_func(F.conv2d, _conv_flops_compute)
    F.conv3d = _wrap_func(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = _wrap_func(F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = _wrap_func(F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = _wrap_func(F.conv_transpose3d, _conv_trans_flops_compute)

    # activation
    F.relu = _wrap_func(F.relu, _relu_flops_compute)
    F.prelu = _wrap_func(F.prelu, _prelu_flops_compute)
    F.elu = _wrap_func(F.elu, _elu_flops_compute)
    F.leaky_relu = _wrap_func(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = _wrap_func(F.relu6, _relu6_flops_compute)
    if hasattr(F, "silu"):
        F.silu = _wrap_func(F.silu, _silu_flops_compute)
    F.gelu = _wrap_func(F.gelu, _gelu_flops_compute)

    # Normalizations
    F.batch_norm = _wrap_func(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = _wrap_func(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = _wrap_func(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = _wrap_func(F.group_norm, _group_norm_flops_compute)

    # poolings
    F.avg_pool1d = _wrap_func(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = _wrap_func(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = _wrap_func(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = _wrap_func(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = _wrap_func(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = _wrap_func(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = _wrap_func(F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = _wrap_func(F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = _wrap_func(F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = _wrap_func(F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = _wrap_func(F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = _wrap_func(F.adaptive_max_pool3d, _pool_flops_compute)

    # upsample
    F.upsample = _wrap_func(F.upsample, _upsample_flops_compute)
    F.interpolate = _wrap_func(F.interpolate, _upsample_flops_compute)

    # softmax
    F.softmax = _wrap_func(F.softmax, _softmax_flops_compute)

    # embedding
    F.embedding = _wrap_func(F.embedding, _embedding_flops_compute)
    if has_flash_attn:
        fa.flash_attn_unpadded_qkvpacked_func = _wrap_func(
            fa.flash_attn_unpadded_qkvpacked_func, _flash_attn_flops_compute
        )


def _reload_functionals():
    # FC
    F.linear = old_functions[F.linear.__name__]

    # convolutions
    F.conv1d = old_functions[F.conv1d.__name__]
    F.conv2d = old_functions[F.conv2d.__name__]
    F.conv3d = old_functions[F.conv3d.__name__]

    # conv transposed
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__name__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__name__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__name__]

    # activation
    F.relu = old_functions[F.relu.__name__]
    F.prelu = old_functions[F.prelu.__name__]
    F.elu = old_functions[F.elu.__name__]
    F.leaky_relu = old_functions[F.leaky_relu.__name__]
    F.relu6 = old_functions[F.relu6.__name__]
    if hasattr(F, "silu"):
        F.silu = old_functions[F.silu.__name__]
    F.gelu = old_functions[F.gelu.__name__]

    # Normalizations
    F.batch_norm = old_functions[F.batch_norm.__name__]
    F.layer_norm = old_functions[F.layer_norm.__name__]
    F.instance_norm = old_functions[F.instance_norm.__name__]
    F.group_norm = old_functions[F.group_norm.__name__]

    # poolings
    F.avg_pool1d = old_functions[F.avg_pool1d.__name__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__name__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__name__]
    F.max_pool1d = old_functions[F.max_pool1d.__name__]
    F.max_pool2d = old_functions[F.max_pool2d.__name__]
    F.max_pool3d = old_functions[F.max_pool3d.__name__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__name__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__name__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__name__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__name__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__name__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__name__]

    # unsample
    F.upsample = old_functions[F.upsample.__name__]
    F.interpolate = old_functions[F.interpolate.__name__]

    # softmax
    F.softmax = old_functions[F.softmax.__name__]

    # embedding
    F.embedding = old_functions[F.embedding.__name__]


def _patch_tensor_methods():
    torch.matmul = _wrap_func(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = _wrap_func(torch.Tensor.matmul, _matmul_flops_compute)
    torch.mm = _wrap_func(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = _wrap_func(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = _wrap_func(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = _wrap_func(torch.bmm, _matmul_flops_compute)

    torch.addmm = _wrap_func(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = _wrap_func(torch.Tensor.addmm, _tensor_addmm_flops_compute)

    torch.mul = _wrap_func(torch.mul, _mul_flops_compute)
    torch.Tensor.mul = _wrap_func(torch.Tensor.mul, _mul_flops_compute)

    torch.add = _wrap_func(torch.add, _add_flops_compute)
    torch.Tensor.add = _wrap_func(torch.Tensor.add, _add_flops_compute)

    torch.einsum = _wrap_func(torch.einsum, _einsum_flops_compute)

    torch.Tensor.reshape = _wrap_func(torch.Tensor.reshape, _pool_flops_compute)
    torch.Tensor.unflatten = _wrap_func(torch.Tensor.unflatten, _pool_flops_compute)
    torch.Tensor.permute = _wrap_func(torch.Tensor.permute, _pool_flops_compute)
    torch.Tensor.view = _wrap_func(torch.Tensor.view, _pool_flops_compute)


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__name__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__name__]
    torch.mm = old_functions[torch.mm.__name__]
    torch.Tensor.mm = old_functions[torch.mm.__name__]
    torch.bmm = old_functions[torch.bmm.__name__]
    torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__name__]

    torch.addmm = old_functions[torch.addmm.__name__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__name__]
    torch.mul = old_functions[torch.mul.__name__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__name__]
    torch.add = old_functions[torch.add.__name__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__name__]

    torch.einsum = old_functions[torch.einsum.__name__]
    torch.Tensor.reshape = old_functions[torch.Tensor.reshape.__name__]
    torch.Tensor.unflatten = old_functions[torch.Tensor.unflatten.__name__]
    torch.Tensor.permute = old_functions[torch.Tensor.permute.__name__]
    torch.Tensor.view = old_functions[torch.Tensor.view.__name__]


def _get_module_params(module):
    s = module.__params__
    for child in module.children():
        s += _get_module_params(child)
    return s


def _get_module_flops(module):
    s = module.__flops__
    for child in module.children():
        s += _get_module_flops(child)
    return s


def _get_module_macs(module):
    s = module.__macs__
    for child in module.children():
        s += _get_module_macs(child)
    return s


def _get_module_duration(module):
    duration = module.__duration__
    if duration == 0:  # e.g. ModuleList
        for child in module.children():
            duration += child.__duration__
    return duration


def _get_module_activation(module):
    activation = module.__activation__
    if activation == 0:  # e.g. ModuleList
        for child in module.children():
            activation += child.__activation__
    return activation


def _num_to_string(num, precision=2):
    if num // 10**12 > 0:
        return str(round(num / 10.0**12, precision)) + " T"
    if num // 10**9 > 0:
        return str(round(num / 10.0**9, precision)) + " G"
    elif num // 10**6 > 0:
        return str(round(num / 10.0**6, precision)) + " M"
    elif num // 10**3 > 0:
        return str(round(num / 10.0**3, precision)) + " K"
    else:
        return str(num) + ""


def _macs_to_string(macs, precision=2):
    return _num_to_string(macs, precision)


def _flops_to_string(flops, precision=2):
    return _num_to_string(flops, precision)


def _params_to_string(params_num, precision=2):
    return _num_to_string(params_num, precision)


def _activation_to_string(activation, precision=2):
    return _num_to_string(activation, precision) + "B"


def _duration_to_string(duration, precision=2):
    if duration > 1:
        return str(round(duration, precision)) + " s"
    elif duration * 10**3 > 1:
        return str(round(duration * 10**3, precision)) + " ms"
    elif duration * 10**6 > 1:
        return str(round(duration * 10**6, precision)) + " us"
    else:
        return str(duration)


def report_metrics(steps, metrics=None, prof=None, distribute=False, aistudio_log=False):
    """
    Report user custom metrics to aistudio.
    Args:
        steps(int): current training step
        metrics(dict):  key is metric name, value is corresponding value
        prof(optional): Aprofiler instance
        distribute(bool): if each pod report metrics
        aistudio_log(bool): if write to aistudio log file
    """
    if aistudio_log and algo_metrics_utils is None:
        raise ImportError("You need aistudio-common>=0.0.28.30 to use report_metrics()")
    local_rank = atorch.distributed.local_rank()
    if local_rank != 0:
        return
    edl_enabled = strtobool(os.getenv("ELASTICDL_ENABLED", "false"))
    is_node_0 = os.getenv("POD_NAME", "").endswith("worker-0" if edl_enabled else "master-0")
    if not distribute and not is_node_0:
        logger.warning("This pod is not node_0, skip report metrics")
        return
    tensorboard_path = os.getenv("VISUAL_DATA_PATH")
    tensorboard_path = os.path.expandvars(tensorboard_path) if tensorboard_path else None
    if metrics:
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict.")
        logger.info(f"Report metrics: {metrics}")
        if aistudio_log:
            algo_metrics_utils.log_metrics(metrics)
        if tensorboard_path:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=tensorboard_path)
            for k, v in metrics.items():
                if distribute:
                    k = str(k) + "/" + str(atorch.distributed.rank())
                writer.add_scalar(k, v, steps)
            writer.close()
    else:
        logger.warning("No metrics to be reported.")
    if prof:
        if not isinstance(prof, AProfiler):
            raise TypeError("prof must be a instance of Aprofiler.")
        algo_metrics_utils.log_metrics({AnalyserConstants.MODEL_FLOPS: prof.get_total_flops()})
        # TODO: report GPU type、amp_used to aistudio


def get_gpu_name():
    gpu = torch.cuda.get_device_properties(0)
    return gpu.name


def _report_metrics_internal(metrics):
    """
    Report metrics to easydl DB
    Args:
        metrics(dict):  key is metric name, value is corresponding value
    """
    if not _REPORTER_ENABLED:
        return
    _EASYDL_COLLECTOR.report_customized_data(_UNIQUE_JOB_ID, metrics)
