import os
import unittest

import torch

import atorch
from atorch.auto.dry_runner.dry_runner import get_dryrunner
from atorch.tests.test_utils import run_multi_process_init_distributed, start_coverage, stop_coverage
from atorch.tests.toy_module import create_model_context


def two_process_profile_func():
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    data_size = 16
    batch_size = 2
    warmup_step_num = 0
    profile_step_num = 4

    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    model_context = create_model_context(data_size=data_size, batch_size=batch_size)
    if torch.cuda.is_available():
        model_context.model.to("cuda")
    dry_runner = get_dryrunner()
    model_context.update_dataloader()
    model_context.update_optim()
    status, result = dry_runner.profile(
        model_context, warmup_step_num=warmup_step_num, profile_step_num=profile_step_num
    )
    assert status
    assert result is not None
    atorch.reset_distributed()


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
        code_path = os.path.abspath(__file__)
        run_multi_process_init_distributed(nproc=2, training_script=code_path)


if __name__ == "__main__":
    cov_status = start_coverage()
    two_process_profile_func()
    if cov_status:
        stop_coverage()
