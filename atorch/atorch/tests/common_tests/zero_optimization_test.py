import copy
import unittest
from unittest.mock import patch

import torch
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.nn.misc import ParamBucket
from fairscale.optim.oss import OSS
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import atorch
from atorch.auto import auto_accelerate
from atorch.auto.opt_lib.utils import to_module_class_by_name
from atorch.auto.opt_lib.zero_optimization import FSDPOptimization, Zero1Optimization, Zero2Optimization
from atorch.tests.toy_modules.toy_module import ToyCustomModule, create_model_context
from atorch.utils.patch_fairscale import patch_add_param_as_view, patch_setup_flat_buffers
from atorch.utils.version import torch_version


class ZeroOptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_context = create_model_context(use_custom_module=True)

    def test_zero_optimization(self):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        atorch.init_distributed(backend)
        zero1_optimization = Zero1Optimization()
        zero2_optimization = Zero2Optimization()
        fsdp_optimization = FSDPOptimization()

        # test zero1
        zero1_model_context = copy.deepcopy(self.model_context)
        zero1_model_context = zero1_optimization.transform(zero1_model_context, config=None)
        self.assertIsInstance(zero1_model_context.create_optim(), OSS)

        # test zero2
        zero2_model_context = copy.deepcopy(self.model_context)
        zero2_model_context = zero2_optimization.transform(zero2_model_context, None)
        zero2_model_context.apply_wrappers(is_pre_wrapper=True)
        zero2_model_context.update_optim()
        zero2_model_context.apply_wrappers(is_pre_wrapper=False)
        if torch_version() < (1, 12, 0) or not torch.cuda.is_available():
            self.assertIsInstance(zero2_model_context.optim, OSS)
            self.assertIsInstance(zero2_model_context.model, ShardedDDP)
        else:
            self.assertIsInstance(zero2_model_context.model, FSDP)

        # test zero2 with not_use_fsdp
        zero2_model_context = copy.deepcopy(self.model_context)
        zero_conf = {"sync_models_at_startup": True, "not_use_fsdp": True}
        zero2_model_context = zero2_optimization.transform(zero2_model_context, config=zero_conf)
        zero2_model_context.update_optim()
        zero2_model_context.apply_wrappers(is_pre_wrapper=False)
        self.assertTrue("zero2" in zero2_model_context.post_wrappers)
        self.assertIsInstance(zero2_model_context.model, ShardedDDP)

        # test fsdp, gpu only
        if torch.cuda.is_available():
            fsdp_model_context = copy.deepcopy(self.model_context)
            zero_conf = {"sync_module_states": True, "forward_prefetch": True, "atorch_wrap_cls": ("Linear",)}
            fsdp_model_context = fsdp_optimization.transform(fsdp_model_context, config=zero_conf)
            fsdp_model_context.apply_wrappers(is_pre_wrapper=True)
            self.assertTrue("fsdp" in fsdp_model_context.pre_wrappers)
            self.assertIsInstance(fsdp_model_context.model, FSDP)
        atorch.reset_distributed()

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_fsdp_wrap_trainable_outmost(self):
        def run_wrap_trainable_outmost(config):
            fsdp_model_context = copy.deepcopy(self.model_context)
            fsdp_optimization = FSDPOptimization()
            for name, param in fsdp_model_context.model.named_parameters():
                if "bias" in name:
                    param.requires_grad = False
            atorch.init_distributed("nccl")
            zero_conf = {
                "sync_module_states": True,
                "forward_prefetch": True,
                "atorch_wrap_cls": ("Linear",),
                "wrap_trainable_outmost": config,
            }
            fsdp_model_context = fsdp_optimization.transform(fsdp_model_context, config=zero_conf)
            if torch_version() < (2, 1, 0):
                with self.assertRaises(RuntimeError):
                    fsdp_model_context.apply_wrappers(is_pre_wrapper=True)
            else:
                fsdp_model_context.apply_wrappers(is_pre_wrapper=True)
                self.assertIsInstance(fsdp_model_context.model, FSDP)
                cpu_device = torch.device("cpu")
                for _, m in fsdp_model_context.model.named_modules():
                    if isinstance(m, FSDP):
                        self.assertTrue(len(m._ignored_params) == 0)
                trainable_param_num = 0
                for p in fsdp_model_context.model.parameters():
                    self.assertTrue(p.device != cpu_device)
                    if p.requires_grad:
                        trainable_param_num += 1
                self.assertEqual(trainable_param_num, 1)
            atorch.reset_distributed()

        configs = [True, "NO_SHARD"]
        for config in configs:
            run_wrap_trainable_outmost(config)

    def test_to_module_class_by_name(self):
        model = self.model_context.model
        wrap_cls = [ToyCustomModule]
        result = to_module_class_by_name(model, wrap_cls)
        self.assertEqual(len(result), 1)
        wrap_cls = ["ToyCustomModule"]
        result = to_module_class_by_name(model, wrap_cls)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ToyCustomModule)
        wrap_cls = (ToyCustomModule, "Linear", "bad_module")
        result = to_module_class_by_name(model, wrap_cls)
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result[0], ToyCustomModule)
        self.assertEqual(result[1].__name__, "Linear")

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    def test_fsdp(self):
        model_context = create_model_context(data_size=4, batch_size=1)
        # test fsdp in cpu mode
        from atorch.common.log_utils import default_logger as logger

        old_info_fn = logger.info

        def new_info_fn(*args, **kwargs):
            old_info_fn(*args, **kwargs)

        with patch.object(logger, "info", side_effect=new_info_fn) as mock_obj:
            auto_accelerate(
                model_context.model,
                load_strategy=[
                    (
                        "fsdp",
                        {
                            "atorch_wrap_cls": (nn.Linear,),
                        },
                    )
                ],
            )
        found = False
        for call_args_kwargs in mock_obj.call_args_list:
            args = call_args_kwargs[0]
            if args[0] == ("These distributed optimization methods are ignored in non-distributed case: %s"):
                found = True
                self.assertListEqual(args[1], ["fsdp"])
                break
        self.assertTrue(found)

    def test_deepcopy(self):
        model_context = create_model_context(data_size=4, batch_size=1)
        model = model_context.model
        need_ignore_modulse = [model.linear]
        # using deepcopy will cause ignore_modules no work
        ignore_modules = copy.deepcopy(need_ignore_modulse)
        found = False
        # simulate the behavior of _recursive_wrap
        for _, child in model.named_modules():
            if child in ignore_modules:
                found = True
        self.assertFalse(found)

        ignore_modules = copy.copy(need_ignore_modulse)
        found = False
        for _, child in model.named_modules():
            if child in ignore_modules:
                found = True
        self.assertTrue(found)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch_version() == (2, 0, 0),
        "Skip when torch==2.0.0. It may be a bug of torch==2.0.0",
    )
    def test_cpu_offload(self):
        backend = "nccl"
        atorch.init_distributed(backend)
        model_context = create_model_context(data_size=4, batch_size=1)
        fsdp_opt = FSDPOptimization()
        model_context = fsdp_opt.transform(model_context)
        model_context.update_dataloader()
        model_context.update_optim()
        config = {"cpu_offload": True}
        # cpu offload can be used in non-distributed mode
        self.assertFalse(fsdp_opt.distributed_only(config=config))
        fsdp_opt.apply_wrapper(model_context, "", wrapper_config=config)
        for param in model_context.model.parameters():
            self.assertEqual(param.device.type, "cpu")
        atorch.reset_distributed()

    def test_fairscale_patch(self):
        self.assertEqual(OSS._setup_flat_buffers, patch_setup_flat_buffers)
        self.assertEqual(ParamBucket._add_param_as_view, patch_add_param_as_view)


if __name__ == "__main__":
    unittest.main()
