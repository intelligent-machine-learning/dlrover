import functools
from copy import deepcopy

import torch
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as auto_wrap_policy
except ImportError:
    from torch.distributed.fsdp.wrap import default_auto_wrap_policy as auto_wrap_policy

from atorch.auto.opt_lib.optimization import Optimization
from atorch.auto.opt_lib.utils import find_modules, to_module_class_by_name
from atorch.distributed.distributed import local_rank, parallel_group, parallel_group_size
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.utils.meta_model_utils import is_meta
from atorch.utils.version import torch_version


class Zero1Optimization(Optimization):
    def __init__(self):
        super().__init__(name="zero1", group="zero", is_tunable=False, is_distributed=True)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        """Transform optimizer use Fairscale Zero1
        Args:
            model_context: ModelContext instance
            config(dict): config for Fairscale Zero1
        Returns:
            transformed ModelContext instance
        """
        # skip zero1 optimization when user did not pass optim_func
        if model_context.optim_func is None:
            return model_context
        new_optim_args = {}
        new_optim_args["optim"] = model_context.optim_func
        new_optim_args["group"] = parallel_group("data")
        new_optim_args.update(model_context.optim_args)

        model_context.optim_func = OSS
        model_context.optim_args = new_optim_args

        return model_context


class Zero2Optimization(Optimization):
    def __init__(self):
        super().__init__(name="zero2", group="zero", is_tunable=False, is_distributed=True)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        """Transform optimizer use Fairscale Zero2
        Args:
            model_context: ModelContext instance
            config(dict): config for Fairscale Zero2
        Returns:
            transformed ModelContext instance
        """
        # skip zero2 optimization when user did not pass optim_func
        if model_context.optim_func is None:
            return model_context
        config = deepcopy(config) or {}
        not_use_fsdp = config.pop("not_use_fsdp", False)
        if not_use_fsdp or torch_version() < (1, 12, 0) or not torch.cuda.is_available():
            # use fairscale zero2 with OSS
            new_optim_args = {}
            new_optim_args["optim"] = model_context.optim_func
            new_optim_args["group"] = parallel_group("zero") or parallel_group("data")
            new_optim_args.update(model_context.optim_args)

            model_context.optim_func = OSS
            model_context.optim_args = new_optim_args
            model_context.add_wrapper(
                "zero2",
                Zero2Optimization.apply_wrapper,
                wrapper_config=config,
                is_pre_wrapper=False,
            )
        else:
            # use fsdp zero2
            from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

            config["sharding_strategy"] = ShardingStrategy.SHARD_GRAD_OP
            model_context.add_wrapper(
                "fsdp",
                FSDPOptimization.apply_wrapper,
                wrapper_config=config,
                is_pre_wrapper=True,
            )

        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        """Zero2 must be used after optimizer is created, it's a post wrapper"""
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
            device = torch.device(type="cuda", index=local_rank())
            model_context.model.to(device)
        wrapper_config = wrapper_config or {}
        mixed_with_ddp = parallel_group_size("data") and parallel_group_size("data") > 1 and parallel_group("zero")

        if torch_version() < (1, 12, 0) and mixed_with_ddp:
            raise ValueError("Zero + ddp only support pytorch 1.12.0 or later.")
        model_context.model = ShardedDDP(
            model_context.model,
            model_context.optim,
            parallel_group("zero") or parallel_group("data"),
            **wrapper_config,
        )

        return model_context


class FSDPOptimization(Optimization):
    def __init__(self):
        super().__init__(name="fsdp", group="zero", is_tunable=False, is_distributed=True)

    def distributed_only(self, config=None):
        # cpu offload can be used in non-distributed mode
        if config is not None and config.get("cpu_offload", False) is True:
            return False
        return True

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        """Transform use FSDP
        Args:
            model_context: ModelContext instance
            config(dict): FSDP parameters and optional atorch specific configs below.
                          atorch_wrap_cls:
                              tuple/list of module classes to wrap with fsdp.
                          atorch_ignored_cls:
                              tuple of module classes, modules whose are instances of these classes would be ignored.
                          atorch_size_based_min_num_params (default 1e5):
                              if atorch_wrap_cls not exist, use size_based_auto_wrap_policy with this min_num_params.
        Returns:
            transformed ModelContext instance
        """
        if not torch.cuda.is_available():
            raise ValueError("FSDP only support GPU !")
        model_context.add_wrapper(
            "fsdp", FSDPOptimization.apply_wrapper, wrapper_config=deepcopy(config), is_pre_wrapper=True
        )

        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        """FSDP must be created before optimizer is created, it's a pre wrapper"""
        torch.cuda.set_device(local_rank())
        wrapper_config = wrapper_config or {}
        # atorch_wrap_cls or atorch_size_based_min_num_params
        if "atorch_wrap_cls" in wrapper_config and torch_version() >= (1, 12, 1):
            wrap_cls = wrapper_config["atorch_wrap_cls"]
            del wrapper_config["atorch_wrap_cls"]
            # atorch_wrap_cls may contain string for module name, convert to module class.
            wrap_cls = to_module_class_by_name(model_context.model, wrap_cls)
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=set(wrap_cls))
            wrapper_config["auto_wrap_policy"] = wrap_policy
        else:
            policy_param_name = "auto_wrap_policy" if torch_version() >= (1, 12, 0) else "fsdp_auto_wrap_policy"
            if "atorch_size_based_min_num_params" in wrapper_config:
                min_num_params = wrapper_config["atorch_size_based_min_num_params"]
                del wrapper_config["atorch_size_based_min_num_params"]
            else:
                min_num_params = 1e5
            wrapper_config[policy_param_name] = functools.partial(auto_wrap_policy, min_num_params=min_num_params)

        if torch_version() >= (1, 12, 0):
            # ignore embedding
            if "atorch_ignored_cls" in wrapper_config:
                ignored_cls = wrapper_config["atorch_ignored_cls"]
                del wrapper_config["atorch_ignored_cls"]
                ignored_cls = to_module_class_by_name(model_context.model, ignored_cls)
                ignore_modules = find_modules(model_context.model, ignored_cls)
                if ignore_modules:
                    wrapper_config["ignored_modules"] = set(ignore_modules)
            # default to use "backward_prefetch"
            if "backward_prefetch" not in wrapper_config:
                from torch.distributed.fsdp import BackwardPrefetch

                wrapper_config["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            cpu_offload = wrapper_config.get("cpu_offload", False)
            if not cpu_offload:
                # Initialize modules' params on gpu and set device id
                # support meta
                if is_meta(model_context.model):
                    wrapper_config["param_init_fn"] = functools.partial(
                        materialize_modules_to_device, device=local_rank()
                    )
                # set device to gpu
                wrapper_config.setdefault("device_id", local_rank())
            else:
                if "sync_module_states" in wrapper_config:
                    # cpu_offload do not support sync_module_states
                    wrapper_config.pop("sync_module_states")
                from torch.distributed.fsdp import CPUOffload

                wrapper_config["cpu_offload"] = CPUOffload(offload_params=True)

        fsdp_clz = FSDP
        pg = parallel_group("zero") or parallel_group("data")
        extra_config = {}
        hybrid_with_ddp = (
            parallel_group_size("zero")
            and parallel_group_size("data")
            and parallel_group_size("zero") > 1
            and parallel_group_size("data") > 1
        )
        if hybrid_with_ddp:
            if torch_version() == (1, 12, 1):
                from atorch.data_parallel.zero_ddp_mix_112 import FSDPWithDDP

                fsdp_clz = FSDPWithDDP
            elif torch_version() < (2, 0):
                raise ValueError(f"Pytorch version {torch_version()} does not support FSDP + DDP hybrid sharding.")
            else:
                # pytorch version >= (2, 0), use fsdp HYBRID_SHARD
                from torch.distributed.fsdp.api import ShardingStrategy

                extra_config["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
                pg = (parallel_group("zero"), parallel_group("data"))

        if wrapper_config.pop("parse_fsdp", None) is not None:
            cb = wrapper_config.pop("parse_cb", None)
            from atorch.utils.parse_fsdp_mapping import ParseFSDP

            fsdp_clz = ParseFSDP(fsdp_clz, fsdp_clz is FSDP, cb)

        wrapper_config.update(extra_config)
        model_context.model = fsdp_clz(
            model_context.model,
            pg,
            **wrapper_config,
        )
        if torch_version() < (1, 12, 0):
            model_context.model.to(local_rank())
        return model_context
