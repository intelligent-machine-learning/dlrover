import unittest
from string import Template

import torch

from atorch.auto.dry_runner.dry_runner import get_dryrunner
from atorch.tests.test_utils import run_multi_process_init_distributed
from atorch.tests.toy_module import create_model_context

code_template = Template(
    """
import atorch
from atorch.auto.model_context import ModelContext
from atorch.auto.dry_runner.dry_runner import get_dryrunner
from atorch.tests.toy_module import create_model_context
from atorch.auto.accelerate import run_task
import torch

if __name__ == "__main__":
    res = atorch.init_distributed("$backend", set_cuda_device_using_local_rank=True, $kargs)
    if not res:
        raise Exception("init failed")
    model_context = create_model_context(data_size=$data_size, batch_size=$batch_size)
    if torch.cuda.is_available():
        model_context.model.to("cuda")
    dry_runner = get_dryrunner()
    model_context.update_dataloader()
    model_context.update_optim()
    status, result = dry_runner.profile(model_context, warmup_step_num=$warmup_step_num,
                                        profile_step_num=$profile_step_num)
    assert status
    assert result is not None
    atorch.reset_distributed()
"""
)


class DryRunnerTest(unittest.TestCase):
    def test_one_process_profile(self):
        data_size = 8
        batch_size = 2
        model_context = create_model_context(data_size=data_size, batch_size=batch_size)
        if torch.cuda.is_available():
            model_context.model.to("cuda")
        model_context.update_dataloader()
        model_context.update_optim()
        dry_runner = get_dryrunner()
        status, result = dry_runner.profile(model_context, warmup_step_num=1, profile_step_num=2)
        self.assertTrue(status)
        self.assertTrue(result is not None)
        status, result = dry_runner.profile(model_context, warmup_step_num=0, profile_step_num=2)
        self.assertTrue(status)
        self.assertTrue(result is not None)
        status, result = dry_runner.profile(model_context, warmup_step_num=0, profile_step_num=200)
        self.assertTrue(not status)

    def test_two_process_profile(self):
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        data_size = 16
        batch_size = 2
        kargs = ""
        warmup_step_num = 0
        profile_step_num = 4

        codes = code_template.substitute(
            backend=backend,
            kargs=kargs,
            data_size=data_size,
            batch_size=batch_size,
            warmup_step_num=warmup_step_num,
            profile_step_num=profile_step_num,
        )
        run_multi_process_init_distributed(codes, nproc=2)
