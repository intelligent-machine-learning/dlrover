import copy
import unittest

import torch
from fairscale.optim.oss import OSS

import atorch
from atorch.distributed.distributed import reset_distributed
from atorch.optimizers import BF16Optimizer


class BF16OptimizerTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "run on gpu")
    def test_bf16_optimizer(self):

        model = torch.nn.Linear(2, 2).bfloat16().to(0)
        model1 = copy.deepcopy(model)
        inputs = torch.ones(2, 2).bfloat16().to(0)
        mse_loss = torch.nn.MSELoss()
        label = torch.ones(2, 2).bfloat16().to(0)
        y = model(inputs)
        loss = mse_loss(y, label)
        loss.backward()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        bf_16_grad = copy.deepcopy([p.grad for p in model.parameters()])
        bf_16_params_after_optimizer_step = copy.deepcopy([p for p in model.parameters()])
        optimizer.zero_grad()

        inputs = torch.ones(2, 2).bfloat16().to(0)
        mse_loss = torch.nn.MSELoss()
        label = torch.ones(2, 2).bfloat16().to(0)
        y = model1(inputs)
        loss = mse_loss(y, label)
        loss.backward()
        optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)
        bf_16_finetune_optimizer = BF16Optimizer(optimizer)
        self.assertEqual(len(bf_16_finetune_optimizer.fp32_from_fp16_groups), 1)
        self.assertEqual(len(bf_16_finetune_optimizer.fp32_from_fp32_groups), 1)
        self.assertEqual(len(bf_16_finetune_optimizer.fp16_groups), 1)
        for i, param_group in enumerate(bf_16_finetune_optimizer.optimizer.param_groups):
            for p in param_group["params"]:
                self.assertTrue(p.dtype == torch.float)

        model_grad = copy.deepcopy([p.grad for p in model1.parameters()])
        res = [(p1 - p2).sum().abs() for p1, p2 in zip(model_grad, bf_16_grad)]
        self.assertTrue(sum(res) < 1e-8)
        # self.assertEqual()

        bf_16_finetune_optimizer._model_grads_to_master_grads()
        self.assertEqual(bf_16_finetune_optimizer.fp32_from_fp16_groups[0][0].grad.dtype, torch.float32)

        bf_16_finetune_optimizer.step()
        bf_16_params_bf_16_finetune_optimizer_step = copy.deepcopy([p for p in model.parameters()])
        res = [
            (p1 - p2).sum().abs()
            for p1, p2 in zip(bf_16_params_bf_16_finetune_optimizer_step, bf_16_params_after_optimizer_step)
        ]
        self.assertTrue(sum(res) < 1e-8)
        bf_16_finetune_optimizer.zero_grad()
        self.assertEqual(bf_16_finetune_optimizer.fp32_from_fp16_groups[0][0].grad.sum(), 0)
        self.assertEqual(sum([p.grad.sum() for p in model1.parameters()]), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "run on gpu")
    def test_bf16_finetune_oss_optimizer(self):
        atorch.init_distributed("nccl")

        model = torch.nn.Linear(2, 2).bfloat16().to(0)
        model1 = copy.deepcopy(model)
        inputs = torch.ones(2, 2).bfloat16().to(0)
        mse_loss = torch.nn.MSELoss()
        label = torch.ones(2, 2).bfloat16().to(0)
        y = model(inputs)
        loss = mse_loss(y, label)
        loss.backward()

        oss_optimizer = OSS([p for p in model.parameters()], lr=0.1)

        bf_16_finetune_optimizer = BF16Optimizer(oss_optimizer)
        lr = torch.optim.lr_scheduler.ExponentialLR(bf_16_finetune_optimizer, gamma=0.98)
        self.assertEqual(len(bf_16_finetune_optimizer.fp32_from_fp16_groups), 1)
        self.assertEqual(len(bf_16_finetune_optimizer.fp32_from_fp32_groups), 1)
        self.assertEqual(len(bf_16_finetune_optimizer.fp16_groups), 1)

        for i, param_group in enumerate(bf_16_finetune_optimizer.optimizer.optim.param_groups):
            for p in param_group["params"]:
                self.assertTrue(p.dtype == torch.float)

        for i, param_group in enumerate(bf_16_finetune_optimizer.param_groups):
            for p in param_group["params"]:
                self.assertTrue(p.dtype == torch.float)

        bf_16_finetune_optimizer.step()
        lr.step()

        model_fp32_param_grad = []
        for i, param_group in enumerate(bf_16_finetune_optimizer.optimizer.optim.param_groups):
            for p in param_group["params"]:
                self.assertTrue(p.requires_grad)
                model_fp32_param_grad.append(p.grad)
                self.assertTrue(p.grad.dtype == torch.float)
                self.assertTrue(p.dtype == torch.float)
        y = model1(inputs)
        loss = mse_loss(y, label)
        loss.backward()
        model1_grad = [p.grad for p in model1.parameters()]

        res = [(p1.float() - p2).sum() for p1, p2 in zip(model1_grad, model_fp32_param_grad)]
        self.assertEqual(sum(res), 0)
        res = [p.grad.sum() for p in model.parameters()]
        self.assertTrue(sum(res) != 0)
        bf_16_finetune_optimizer.zero_grad()
        res = [p.grad.sum() for p in model.parameters()]
        self.assertEqual(sum(res), 0)
        reset_distributed()


if __name__ == "__main__":
    unittest.main()
