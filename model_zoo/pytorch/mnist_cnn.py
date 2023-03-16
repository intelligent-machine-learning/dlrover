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

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dlrover.python.elastic_agent.pytorch.elastic_dataset import ElasticDataset


class ElasticMnistDataset(ElasticDataset):
    def __init__(self, path, batch_size, epochs, shuffle):
        """The dataset supports elastic training.

        Args:
            images: A list with tuples like (image_path, label_index).
            For example, we can use `torchvision.datasets.ImageFolder`
            to get the list.
            data_shard_service: If we want to use elastic training, we
            need to use the `data_shard_service` of the elastic controller
            in elasticai_api.
        """
        super(ElasticMnistDataset, self).__init__(
            path,
            batch_size,
            epochs,
            shuffle,
        )

    def read_sample(self, index):
        image_meta = self.lines[index]
        image_path, label = image_meta.split(",")
        image = cv2.imread(image_path)
        image = np.array(image / 255.0, np.float32)
        image = image.reshape(3, 28, 28)
        label = float(label)
        return image, label


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


def train(args):
    """The function to run the training loop.
    Args:
        dataset: The dataset is provided by ElasticDL for the elastic training.
        Now, the dataset if tf.data.Dataset and we need to convert
        the data in dataset to torch.tensor. Later, ElasticDL will
        pass a torch.utils.data.DataLoader.
        elastic_controller: The controller for elastic training.
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Use cuda = f{use_cuda}")
    if use_cuda:
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    local_rank = os.environ["LOCAL_RANK"]
    print(f"rank {rank} is initialized local_rank = {local_rank}")
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = ElasticMnistDataset(
        path=args.training_data,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        shuffle=args.shuffle,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_workers=2
    )

    test_dataset = ElasticMnistDataset(
        path=args.validation_data,
        batch_size=args.batch_size,
        epochs=1,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, num_workers=2
    )

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    if torch.cuda.is_available():
        rank = int(os.environ["LOCAL_RANK"])
        print(f"Running basic DDP example on rank {rank}.")
        # create model and move it to GPU with id rank
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    epoch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        loss = train_one_batch(model, optimizer, data, target)
        print("loss = {}, step = {}".format(loss, batch_idx))
        new_epoch = train_dataset.get_epoch()
        if new_epoch and new_epoch > epoch:
            epoch = new_epoch
            # Set epoch of the scheduler
            scheduler.last_epoch = epoch - 1
            scheduler.step()
            test(model, device, test_loader)
        train_dataset.report_batch_done()


def train_one_batch(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--num_epochs", type=int, default=1, required=False)
    parser.add_argument("--shuffle", type=bool, default=False, required=False)
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
