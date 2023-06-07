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
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dlrover.trainer.torch.elastic import ElasticTrainer, set_master_addr
from dlrover.trainer.torch.elastic_sampler import ElasticDistributedSampler

CHEKPOINT_PATH = "model.pt"


def log_rank0(msg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(msg)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def cleanup():
    dist.destroy_process_group()


def setup():
    use_cuda = torch.cuda.is_available()
    set_master_addr()
    if use_cuda:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    local_rank = os.environ["LOCAL_RANK"]
    print(f"rank {rank} is initialized local_rank = {local_rank}")


def train(args):
    """The function to run the training loop.
    Args:
        dataset: The dataset is provided by ElasticDL for the elastic training.
        Now, the dataset if tf.data.Dataset and we need to convert
        the data in dataset to torch.tensor. Later, ElasticDL will
        pass a torch.utils.data.DataLoader.
        elastic_controller: The controller for elastic training.
    """
    setup()

    train_data = torchvision.datasets.ImageFolder(
        root=args.training_data,
        transform=transforms.ToTensor(),
    )
    #  Setup sampler for elastic training.
    sampler = ElasticDistributedSampler(dataset=train_data)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=2,
        sampler=sampler,
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=args.validation_data,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    model = Net()

    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Running basic DDP example on local rank {local_rank}.")
        # create model and move it to GPU with id rank
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
        print(f"Model device {model.device}")
    else:
        model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    checkpoint = load_checkpoint(CHEKPOINT_PATH)
    if checkpoint:
        model.load_state_dict(checkpoint.get("model", {}))
        optimizer.load_state_dict(checkpoint.get("optimizer", {}))
        #  Restore sampler from checkpoint.
        train_loader.sampler.load_state_dict(checkpoint.get("sampler", {}))

    epochs = args.num_epochs
    if args.fixed_batch_size:
        train_with_fixed_batch_size(
            model,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
            epochs,
        )
    else:
        train_with_elastic_batch_size(
            model,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
            epochs,
        )


def train_with_fixed_batch_size(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    epochs,
):
    """
    The global batch size will not change if the number of workers changes.
    """
    elastic_trainer = ElasticTrainer(model)
    optimizer, scheduler = elastic_trainer.prepare(optimizer, scheduler)

    epoch = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    start_epoch = train_loader.sampler.epoch
    for epoch in range(start_epoch, epochs):
        elastic_trainer.reset()
        for _, (data, target) in enumerate(train_loader):
            model.train()
            with elastic_trainer.step():
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_step = elastic_trainer.num_steps
                if train_step % 20 == 0:
                    print("loss = {}, step = {}".format(loss, train_step))

                if train_step > 0 and train_step % 200 == 0:
                    print("Save checkpoint.")
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sampler": train_loader.sampler.state_dict(
                            train_step, train_loader.batch_size
                        ),  # Checkpoint sampler
                    }
                    torch.save(checkpoint, CHEKPOINT_PATH)
        scheduler.step()
        print("Test model after epoch {}".format(epoch))
        test(model, device, test_loader)


def train_with_elastic_batch_size(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    epochs,
):
    """The global batch size will change if the number of worker changes."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    start_epoch = train_loader.sampler.epoch
    for epoch in range(start_epoch, epochs):
        train_loader.sampler.set_epoch(epoch)
        for step, (data, target) in enumerate(train_loader):
            model.train()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 20 == 0:
                log_rank0("loss = {}, step = {}".format(loss, step))
            if step > 0 and step % 200 == 0:
                log_rank0("Save checkpoint.")
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sampler": train_loader.sampler.state_dict(
                        step, train_loader.batch_size
                    ),  # Checkpoint sampler
                }
                torch.save(checkpoint, CHEKPOINT_PATH)
        scheduler.step()
        log_rank0("Test model after epoch {}".format(epoch))
        test(model, device, test_loader)


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    checkpoint = torch.load(path)
    return checkpoint


def test(model, device, test_loader):
    log_rank0("Test the model ...")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        model = model.to(device)
        for data, target in test_loader:
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    log_rank0(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


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
