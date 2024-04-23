# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

from dlrover.python.common.multi_process import clear_sock_dir
from dlrover.python.elastic_agent.torch.ckpt_saver import DdpCheckpointSaver
from dlrover.trainer.torch.flash_checkpoint.hf_trainer import (
    FlashCkptTrainer,
    HfDdpCheckpointer,
)


class FlashCkptTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        DdpCheckpointSaver._saver_instance = None
        DdpCheckpointSaver.start_async_saving_ckpt()

    def tearDown(self) -> None:
        if DdpCheckpointSaver._saver_instance:
            DdpCheckpointSaver._saver_instance.close()
        clear_sock_dir()

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_arguments = TrainingArguments(
                learning_rate=0.001,
                optim="adamw_torch",
                per_device_train_batch_size=1,
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=10,
                save_steps=10,
                output_dir=tmpdir,
                save_total_limit=2,
                load_best_model_at_end=True,
                save_safetensors=False,
            )

            config = LlamaConfig(
                hidden_size=16, num_attention_heads=4, num_hidden_layers=2
            )
            model = LlamaForCausalLM(config)
            trainer = FlashCkptTrainer(model, args=training_arguments)

            for step in [10, 20, 30, 40]:
                ckpt_dir = os.path.join(tmpdir, f"checkpoint-{step}")
                os.makedirs(ckpt_dir)
            tracer_file = os.path.join(tmpdir, "dlrover_latest.txt")
            with open(tracer_file, "w") as f:
                f.write("40")
            step = trainer._get_last_checkpoint_step()
            self.assertEqual(step, 40)
            trainer._rotate_checkpoints(use_mtime=False, output_dir=tmpdir)
            checkpoints = os.listdir(tmpdir)
            self.assertEqual(len(checkpoints), 3)

            trainer.flash_checkpointer = HfDdpCheckpointer(tmpdir)
            ckpt_dir = os.path.join(tmpdir, "checkpoint-50")
            trainer._save(ckpt_dir)
            cached_sd = trainer.flash_checkpointer.ckpt_agent.state_dict
            self.assertTrue("pytorch_model.bin" in cached_sd)
