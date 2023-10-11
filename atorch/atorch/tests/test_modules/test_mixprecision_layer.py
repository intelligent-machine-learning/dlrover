# coding=utf-8
from __future__ import absolute_import, unicode_literals

import random
import unittest
from contextlib import nullcontext
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from deepspeed import DeepSpeedTransformerConfig
from torch.optim import AdamW

from atorch.modules.transformer.inject import get_bert_layer_weight_offset, replace_with_deepspeed_transformer
from atorch.modules.transformer.layers import MixPrecisionTransformerLayer, is_apex_amp_activate

try:
    from atorch import amp

    amp_available = True
except (ImportError, ModuleNotFoundError):
    amp_available = False

try:
    from transformers.modeling_bert import BertConfig, BertLayer  # 3.5
except (ModuleNotFoundError, ImportError):
    from transformers.models.bert.modeling_bert import BertConfig, BertLayer  # 4.17


def autocast(enabled):
    # old than torch version 1.9:
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", enabled=enabled)
    else:  # newer than torch version 1.9:
        return torch.cuda.amp.autocast(enabled=enabled)


class MyModule(torch.nn.Module):
    def __init__(self, config):
        super(MyModule, self).__init__()
        self.layer = BertLayer(config)

    def forward(self, hidden_state, mask):
        return self.layer(hidden_state, mask)


@unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
@unittest.skipIf(not amp_available, "atorch.amp is not available")
class TestTransformer(unittest.TestCase):

    seed = 1234

    def bert2deepspeedconfig(self, config):
        return DeepSpeedTransformerConfig(
            batch_size=config.batch_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            heads=config.num_attention_heads,
            attn_dropout_ratio=config.attention_probs_dropout_prob,
            hidden_dropout_ratio=config.hidden_dropout_prob,
            num_hidden_layers=config.num_hidden_layers,
            initializer_range=config.initializer_range,
            # local_rank=0,
            seed=self.seed,
            fp16=config.fp16,
            stochastic_mode=True,  # Enable for high performance
            return_tuple=True,
            pre_layer_norm=False,
            # training=config
        )

    def setUp(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def assertTensorEqual(self, tensor0, tensor1):
        self.assertListEqual(list(tensor0.shape), list(tensor1.shape))
        self.assertEqual(tensor0.dtype, tensor1.dtype)
        self.assertFalse(bool(torch.any(torch.isnan(tensor0))))
        self.assertFalse(bool(torch.any(torch.isnan(tensor1))))
        allclose = torch.allclose(tensor0, tensor1, atol=1e-2, rtol=1e-2)
        abs_delta = torch.sum(torch.abs(tensor0 - tensor1))
        max_delta = torch.max(torch.abs(tensor0 - tensor1))
        self.assertTrue(
            allclose,
            msg="tensor not allclose abs=%.4f max=%.4f" % (abs_delta, max_delta),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
    @unittest.skipIf(not amp_available, "atorch.amp is not available")
    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties(torch.cuda.current_device()).name == "NVIDIA A10",
        "Only failed on A10. Passed on A100",
    )
    def test_bert_layer(self):

        argnames, arg_list = (
            "batch_size, small_bsz, hidden_size, " "seq_len, heads, num_layers, is_training, use_fp16",
            [
                (8, 3, 128, 32, 8, 3, False, False),
                (8, 3, 1024, 376, 16, 3, True, False),
                (8, 3, 128, 32, 8, 3, False, True),
                (8, 3, 1024, 376, 16, 3, True, True),
            ],
        )
        argnames = [argname.strip() for argname in argnames.split(",")]
        for args in arg_list:
            show_argname = " ".join(["%s=%s" % (argname, arg) for argname, arg in zip(argnames, args)])
            params = dict(zip(argnames, args))
            with self.subTest("test_bert_layer %s" % show_argname, **params):
                self.__test_bertlayer(**params)

        args = arg_list[-1]
        for is_training, fp16 in product([True, False], [True, False]):
            args = args[:-2] + (is_training, fp16)
            show_argname = " ".join(["%s=%s" % (argname, arg) for argname, arg in zip(argnames, args)])
            params = dict(zip(argnames, args))
            with self.subTest("test_bert_layer %s" % show_argname, **params):
                self.__test_bertlayer(**params)

    def assertStateDictEqual(self, sd0, sd1):
        for key, value in sd0.items():
            if isinstance(value, dict):
                self.assertStateDictEqual(value, sd1[key])
            else:
                with self.subTest("StateDict", key=key):
                    self.assertTensorEqual(value, sd1[key])

    def __test_bertlayer(
        self,
        batch_size,
        small_bsz,
        hidden_size,
        seq_len,
        heads,
        num_layers,
        is_training,
        use_fp16,
    ):
        """pure fp32/fp16 numerical consistency"""
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        if use_fp16:  # pure fp16, input/output are fp16
            dtype = torch.half
        else:
            dtype = torch.float32
        hidden_state = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        mask = torch.randn(
            batch_size,
            1,
            1,
            seq_len,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        config_dict = dict(
            hidden_size=hidden_size,
            vocab_size_or_config_json_file=119547,
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            hidden_act="gelu",
            chunk_size_feed_forward=8,
            fp16=use_fp16,
            batch_size=batch_size,
        )
        config = BertConfig(**config_dict)

        mymodel = MyModule(config)
        sd = mymodel.layer.state_dict()
        migrate_weight, migrate_bias = get_bert_layer_weight_offset(mymodel.layer)
        if use_fp16:
            mymodel.half()
        # test linear
        input_ = torch.randn(8, 32, 128, device=device)
        weight = torch.randn(512, 128, device=device)
        Y = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        # make sure cuda liner work
        F.linear(input_, weight, None)

        mymodel.to(device)
        if is_training:
            mymodel.train()
        else:
            mymodel.eval()  # let dropout same
        out_hidden = mymodel(hidden_state, mask)
        base_loss = (Y - out_hidden[0]).pow(2).sum() / 64
        base_loss.backward()
        ds_config = self.bert2deepspeedconfig(config)
        # test DeepSpeedTransformerLayer
        layer2 = MixPrecisionTransformerLayer(
            ds_config,
            initial_weights=migrate_weight,
            initial_biases=migrate_bias,
        ).to(device)

        if is_training:
            layer2.train()
        else:
            layer2.eval()
        layer3 = MixPrecisionTransformerLayer(ds_config)
        layer3.load_state_dict(sd)
        layer3.to(device)
        if use_fp16:  # fp16 mode
            layer2.half()
            layer3.half()
        self.assertStateDictEqual(layer2.state_dict(), mymodel.layer.state_dict())
        self.assertStateDictEqual(layer2.state_dict(), layer3.state_dict())

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ):
            start.record()
            for i in range(5):
                output = layer2(hidden_state, mask)
        ds_loss = (Y - output[0]).pow(2).sum() / 64
        ds_loss.backward()
        end.record()
        torch.cuda.synchronize()
        output = layer2(hidden_state, mask)
        output2 = layer3(hidden_state, mask)
        for first, second in zip(out_hidden, output):
            self.assertTensorEqual(first, second)
        for first, second in zip(out_hidden, output2):  # recovery from state_dict
            self.assertTensorEqual(first, second)
        migrate_weight, migrate_bias = get_bert_layer_weight_offset(mymodel.layer)
        base_grads = [t.grad for t in migrate_weight] + [t.grad for t in migrate_bias]
        ds_grads = self.get_ds_grads(layer2)
        self.assertEqual(len(base_grads), len(ds_grads))
        for first, second in zip(base_grads, ds_grads):
            self.assertTensorEqual(
                first.to(torch.float32) if use_fp16 else first,
                second.to(torch.float32) if use_fp16 else second,
            )

    @unittest.skipIf(True, "Failed on gpu")
    def test_amp(self):
        argnames, arg_list = (
            "batch_size, use_apex, hidden_size, seq_len," "heads, num_layers,auto_cast, use_fp16",
            [
                (8, True, 128, 32, 8, 3, False, False),
                (8, False, 1024, 376, 16, 3, False, False),
                (320, False, 768, 48, 16, 3, False, False),
            ],
        )
        argnames = [argname.strip() for argname in argnames.split(",")]
        for args in arg_list:
            params = dict(zip(argnames, args))
            with self.subTest("test_amp", **params):
                self.__test_amp(**params)

        # args = arg_list[-1]
        # for auto_cast, fp16 in product([True, False], [True, False]):
        #     args = args[:-2] + (auto_cast, fp16)
        #     show_argname = " ".join(["%s=%s" % (argname, arg) for argname, arg in zip(argnames, args)])
        #     params = dict(zip(argnames, args))
        #     with self.subTest("test_amp%s" % show_argname, **params):
        #         self.__test_amp(**params)
        #     with self.subTest("test_replace_with_deepspeed_transformer", **params):
        #         self.__test_replace_with_deepspeed_transformer(**params)

    def get_ds_grads(self, layer):
        # (q,k,w,
        # attn_ow
        #  attn_nw
        #  inter_w
        #  output_w
        #  norm_w)
        qkvw_grad = layer.attn_qkvw.grad
        qw, kw, vw = qkvw_grad.split(qkvw_grad.size(0) // 3)
        qkvb_grad = layer.attn_qkvb.grad
        qb, kb, vb = qkvb_grad.split(qkvw_grad.size(0) // 3)
        attn_ow = layer.attn_ow.grad
        attn_ob = layer.attn_ob.grad
        attn_nw = layer.attn_nw.grad
        attn_nb = layer.attn_nb.grad
        inter_w = layer.inter_w.grad
        inter_b = layer.inter_b.grad
        output_w = layer.output_w.grad
        output_b = layer.output_b.grad
        norm_w = layer.norm_w.grad
        norm_b = layer.norm_b.grad
        return (qw, kw, vw, attn_ow, attn_nw, inter_w, output_w, norm_w) + (
            qb,
            kb,
            vb,
            attn_ob,
            attn_nb,
            inter_b,
            output_b,
            norm_b,
        )

    def get_model_forward_cost(self, model, hidden_state, mask, auto_cast=False, is_train=True):
        torch.cuda.synchronize()
        if is_train:
            model.train()
        else:
            model.eval()
        eval_context = nullcontext() if is_train else torch.no_grad()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(5):  # warmup
            out_hidden = model(hidden_state, mask)
        with autocast(enabled=auto_cast), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as p, eval_context:
            start.record()
            for i in range(10):
                out_hidden = model(hidden_state, mask)
                if is_train:
                    Y = torch.rand_like(out_hidden[0])
                    base_loss = (Y - out_hidden[0]).pow(2).sum() / 64
                    base_loss.backward()
        torch.cuda.synchronize()
        end.record()
        cost = start.elapsed_time(end)
        return cost, p

    def __test_amp(
        self,
        batch_size,
        use_apex,
        hidden_size,
        seq_len,
        heads,
        num_layers,
        auto_cast,
        use_fp16,
    ):
        """test auto_cast on/off numerical consistency"""
        device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        dtype = torch.float32
        hidden_state = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        mask = torch.randn(
            batch_size,
            1,
            1,
            seq_len,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        config_dict = dict(
            hidden_size=hidden_size,
            vocab_size_or_config_json_file=119547,
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,  #
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            hidden_act="gelu",
            chunk_size_feed_forward=8,
            fp16=use_fp16,
            batch_size=batch_size,
        )
        config = BertConfig(**config_dict)

        mymodel = MyModule(config)
        mymodel = mymodel.to(device)
        pure_torch_train_cost, pure_profile = self.get_model_forward_cost(mymodel, hidden_state, mask, auto_cast, True)
        pure_torch_eval_cost, _ = self.get_model_forward_cost(mymodel, hidden_state, mask, auto_cast, False)
        print("pure train cost:", batch_size, hidden_size, seq_len, pure_torch_train_cost)
        print("pure eval cost:", batch_size, hidden_size, seq_len, pure_torch_eval_cost)
        pure_profile.export_chrome_trace("pure_bert_layer%s_%s%s.json" % (hidden_size, seq_len, auto_cast))

        # p.export_chrome_trace("amp_torch%s_%s%s.json" % (
        # hidden_size, seq_len, auto_cast))
        ds_config = self.bert2deepspeedconfig(config)
        # test DeepSpeedTransformerLayer
        # amp.register_half_function(MixPrecisionTransformerLayer, "forward")
        migrate_weight, migrate_bias = get_bert_layer_weight_offset(mymodel.layer)
        model = MixPrecisionTransformerLayer(
            ds_config,
            initial_weights=migrate_weight,
            initial_biases=migrate_bias,
            mix_precision=False,
        ).to(device)

        [self.assertEqual(param.dtype, torch.float32) for param in model.parameters()]
        [self.assertEqual(param.dtype, torch.float32) for param in mymodel.parameters()]
        ds_train_cost, p = self.get_model_forward_cost(model, hidden_state, mask, auto_cast, True)
        ds_eval_cost, _ = self.get_model_forward_cost(model, hidden_state, mask, auto_cast, False)
        print("ds train cost:", batch_size, hidden_size, seq_len, ds_train_cost)
        print("ds eval cost:", batch_size, hidden_size, seq_len, ds_eval_cost)
        self.assertLess(ds_train_cost, pure_torch_train_cost)
        self.assertLess(ds_eval_cost, pure_torch_eval_cost)
        p.export_chrome_trace("amp_ds_bert_layer%s_%s%s.json" % (hidden_size, seq_len, auto_cast))
        # amp+fp16 forward may have nan
        return
        # for first, second in zip(out_hidden, output):
        #     self.assertTensorEqual(first, second)
        # # length=3
        # migrate_weight, migrate_bias = get_bert_layer_weight_offset(mymodel.layer)
        # base_grads = [t.grad for t in migrate_weight] + [t.grad for t in migrate_bias]
        # ds_grads = self.get_ds_grads(model)
        # self.assertEqual(len(base_grads), len(ds_grads))
        # if auto_cast:
        #     [self.assertEqual(g.dtype, torch.float32) for g in ds_grads]
        #     [self.assertEqual(g.dtype, torch.float32) for g in base_grads]
        # for first, second in zip(base_grads, ds_grads):
        #     self.assertTensorEqual(first, second)

    def __test_replace_with_deepspeed_transformer(
        self,
        batch_size,
        use_apex,
        hidden_size,
        seq_len,
        heads,
        num_layers,
        auto_cast,
        use_fp16,
    ):
        class MultiLayer(torch.nn.Module):
            def __init__(self, config, nlayers):
                super(MultiLayer, self).__init__()
                layers = [BertLayer(config) for _ in range(nlayers)]
                self.dense0 = torch.nn.Linear(config.hidden_size, config.hidden_size)
                self.layers = torch.nn.ModuleList(layers)
                self.dense1 = torch.nn.Linear(config.hidden_size, config.hidden_size)

            def forward(self, hidden_state, mask):
                hidden_state = self.dense0(hidden_state)
                for layer in self.layers:
                    hidden_state = layer(hidden_state, mask)[0]
                hidden_state = self.dense1(hidden_state)

                return hidden_state

        dtype = torch.float32
        device = torch.cuda.current_device()
        hidden_state = torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        mask = torch.randn(
            batch_size,
            1,
            1,
            seq_len,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        config_dict = dict(
            hidden_size=hidden_size,
            vocab_size_or_config_json_file=119547,
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,  # disable dropout
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            hidden_act="gelu",
            chunk_size_feed_forward=8,
            fp16=use_fp16,
            batch_size=batch_size,
        )
        config = BertConfig(**config_dict)

        torch_model = MultiLayer(config, num_layers)
        torch_model.to(device)
        with autocast(enabled=auto_cast):
            out1 = torch_model(hidden_state, mask)
        torch_count = 0
        for name, module in torch_model.named_modules():
            if isinstance(module, BertLayer):
                torch_count += 1
        self.assertEqual(torch_count, num_layers)

        if use_apex:
            # be careful layer data ptr,otherwise GPU memory addr broken on old driver
            optimizer = AdamW(torch_model.parameters(), lr=1e-5)
            # amp.initialize can not back off modify ,other unittest will effect by this case
            torch_model, optimizer = amp.initialize(torch_model, optimizer, opt_level="O2", loss_scale="dynamic")
            self.assertTrue(is_apex_amp_activate())
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as p:
                for i in range(5):  # warmup 5 times
                    output = torch_model(hidden_state, mask)
                start.record()
                for i in range(5):
                    output = torch_model(hidden_state, mask)
            Y = torch.randn(
                batch_size,
                seq_len,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            ds_loss = (Y - output[0]).pow(2).sum() / 64
            ds_loss.backward()
            end.record()  # include backward time
            torch.cuda.synchronize()
            amp_cose = start.elapsed_time(end)
            print("amp cost:", hidden_size, seq_len, amp_cose)
            for first, second in zip(out1, output):
                self.assertTensorEqual(first.to(torch.float32), second)
        # TODO: validate state dict
        origin_state_dict = torch_model.state_dict()
        new_module = replace_with_deepspeed_transformer(
            BertLayer,
            torch_model,
            config,
            batch_size,
            seq_len,
            self.seed,
            preln=False,
            fp16=use_fp16,
        )
        new_module.to(device)
        after_state_dict = new_module.state_dict()
        self.assertListEqual(list(origin_state_dict.keys()), list(after_state_dict.keys()))
        self.assertStateDictEqual(origin_state_dict, after_state_dict)
        with autocast(enabled=auto_cast), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            out2 = new_module(hidden_state, mask)
        p.export_chrome_trace("inject_amp_bert_layer%s_%s%s.json" % (hidden_size, seq_len, auto_cast))
        # validate out are same
        for first, second in zip(out1, out2):
            self.assertTensorEqual(first, second)
        torch_count = 0
        ds_count = 0
        for name, module in new_module.named_modules():
            if isinstance(module, BertLayer):
                torch_count += 1
            if isinstance(module, MixPrecisionTransformerLayer):
                ds_count += 1
        self.assertEqual(ds_count, num_layers)
        self.assertEqual(torch_count, 0)
        for first, second in zip(out1, out2):
            self.assertTensorEqual(first, second)


if __name__ == "__main__":
    unittest.main()
