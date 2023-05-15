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
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dlrover.trainer.torch.elastic import ElasticTrainer
from dlrover.trainer.torch.elastic_dataset import ElasticDataset

CHEKPOINT_PATH = "model.pt"


def build_data_meta(folder):
    """Save the path of sample into a list and we can get the path
    by the index of sample.

    The directory structure of mnist is
    |- root
      |- 0
        |- 1.png
        |- 21.png
      |- 1
        |- 3.png
        |- 6.png
    the meta is a list [
        ("root/0/1.png", 0),
        ("root/0/21.png", 0),
        ("root/3.png", 1),
        ("root/1/6.png", 1),
    ]
    """
    dataset_meta = []
    for d in os.listdir(folder):
        dir_path = os.path.join(folder, d)
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(".png"):
                    file_path = os.path.join(dir_path, f)
                    dataset_meta.append([file_path, d])
    return dataset_meta


class ElasticMnistDataset(ElasticDataset):
    def __init__(self, path, batch_size, epochs, shuffle):
        """The dataset supports elastic training."""
        self.data_meta = build_data_meta(path)
        super(ElasticMnistDataset, self).__init__(
            name="mnist-train",
            dataset_size=len(self.data_meta),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
        )

    def read_sample(self, index):
        image_path, label = self.data_meta[index]
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


def setup(use_cuda):
    print(f"Use cuda = {use_cuda}")
    ElasticTrainer.setup()
    if use_cuda:
        dist.init_process_group("nccl")
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    setup(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint = load_checkpoint(CHEKPOINT_PATH)

    train_dataset = ElasticMnistDataset(
        path=args.training_data,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        shuffle=args.shuffle,
    )
    if checkpoint:
        train_dataset.load_state_dict(checkpoint.get("train_shards", {}))
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_workers=2
    )

    test_dataset = torchvision.datasets.ImageFolder(
        args.validation_data,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

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

    if checkpoint:
        model.load_state_dict(checkpoint.get("model_state_dict", {}))
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))

    if args.fixed_batch_size:
        train_with_fixed_batch_size(
            model, optimizer, scheduler, train_loader, test_loader, device
        )
    else:
        train_with_elastic_batch_size(
            model, optimizer, scheduler, train_loader, test_loader, device
        )


def train_with_fixed_batch_size(
    model, optimizer, scheduler, train_loader, test_loader, device
):
    """
    The global batch size will not change if the number of workers changes.
    """
    elastic_trainer = ElasticTrainer(model)
    optimizer, scheduler = elastic_trainer.prepare(optimizer, scheduler)

    epoch = 0
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
            train_loader.dataset.step()  # Record the batch samples completed.
            print(
                "loss = {}, step = {}".format(loss, elastic_trainer.num_steps)
            )
            if (
                elastic_trainer.num_steps > 0
                and elastic_trainer.num_steps % 200 == 0
            ):
                save_checkpoint(
                    CHEKPOINT_PATH, model, optimizer, train_loader.dataset
                )
            dataset_epoch = train_loader.dataset.get_epoch()
            if dataset_epoch > epoch:
                epoch = dataset_epoch
                scheduler.step()
                test(model, device, test_loader)
    test(model, device, test_loader)


def train_with_elastic_batch_size(
    model, optimizer, scheduler, train_loader, test_loader, device
):
    """The global batch size will change if the number of worker changes."""
    epoch = 0
    for step, (data, target) in enumerate(train_loader):
        model.train()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loader.dataset.step()  # Record the batch samples completed.
        print("loss = {}, step = {}".format(loss, step))
        if step > 0 and step % 200 == 0:
            save_checkpoint(
                CHEKPOINT_PATH, model, optimizer, train_loader.dataset
            )
        dataset_epoch = train_loader.dataset.get_epoch()
        if dataset_epoch > epoch:
            epoch = dataset_epoch
            scheduler.step()
            test(model, device, test_loader)
    test(model, device, test_loader)


def save_checkpoint(path, model, optimizer, dataset: ElasticDataset):
    print("Save checkpoint.")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_shards": dataset.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    checkpoint = torch.load(path)
    return checkpoint


def test(model, device, test_loader):
    print("Test the model ...")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
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
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--num_epochs", type=int, default=1, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument(
        "--fixed-batch-size", type=bool, default=True, required=False
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
