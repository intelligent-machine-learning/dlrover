# Copyright 2023 The DLRover Authors. All rights reserved.
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

import argparse

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dlrover.trainer.torch.elastic_sampler import ElasticDistributedSampler


class UpdateDataStepCallback(pl.Callback):
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        trainer.datamodule.global_step = trainer.global_step

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        trainer.datamodule.train_sampler.set_epoch(trainer.current_epoch)


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size, shuffle):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.global_step = 0
        super().__init__()

    def setup(self, stage):
        # transforms for images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # prepare transforms standard to MNIST
        self.mnist_train = torchvision.datasets.ImageFolder(
            self.train_dir,
            transform=transform,
        )

        self.mnist_test = torchvision.datasets.ImageFolder(
            self.test_dir,
            transform=transform,
        )
        self.train_sampler = ElasticDistributedSampler(
            self.mnist_train,
            shuffle=self.shuffle,
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def state_dict(self):
        steps_per_epoch = len(self.train_dataloader())
        step_in_epoch = self.global_step % steps_per_epoch
        epoch = int(self.global_step / steps_per_epoch)
        self.train_sampler.set_epoch(epoch)
        state = self.train_sampler.state_dict(step_in_epoch, self.batch_size)
        return state

    def load_state_dict(self, state_dict) -> None:
        self.train_sampler.load_state_dict(state_dict)


def train(args):
    data_module = MNISTDataModule(
        args.training_data, args.validation_data, args.batch_size, args.shuffle
    )

    # train
    model = LightningMNISTClassifier()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        every_n_train_steps=1000,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )
    callbacks = [UpdateDataStepCallback(), checkpoint_callback]

    trainer = pl.Trainer(
        max_epochs=2,
        default_root_dir="./data/",
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--num_epochs", type=int, default=1, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument(
        "--fixed_batch_size", type=bool, default=False, required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, required=False
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disable CUDA training",
    )
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument(
        "--validation_data", type=str, default="", required=True
    )
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    train(args)
