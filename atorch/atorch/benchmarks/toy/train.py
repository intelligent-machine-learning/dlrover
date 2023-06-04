import random
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import atorch


def parse_args():
    parser = ArgumentParser(description="training arguments")

    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="The number of epoch",
    )

    parser.add_argument(
        "--data_size",
        type=int,
        default=24800,
        help="The number of training data",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )

    parser.add_argument(
        "--use_gpu",
        nargs="?",
        const=not True,
        default=True,
        type=lambda x: x.lower() in ["true", "yes", "t", "y"],
        help="if use gpu",
    )

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, data_size):
        self.size = data_size
        self.a = 0.1
        self.b = 0.3

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.tensor([random.random()])
        return x, self.a * x + self.b


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


def main(args):
    if args.use_gpu:
        status = atorch.init_distributed("nccl")
    else:
        status = atorch.init_distributed("gloo")

    if status:
        print("int_distribute success")
    else:
        raise Exception("Fail to init distribute")

    # create dataloader for DDP
    dataset = CustomDataset(args.data_size)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, num_workers=2, batch_size=args.batch_size)

    # model
    model = MyModel()
    if args.use_gpu:
        model.to(atorch.local_rank())
        device_ids = [atorch.local_rank()]
        output_device = atorch.local_rank()
    else:
        device_ids = None
        output_device = None

    model = DDP(model, device_ids=device_ids, output_device=output_device)

    # loss func
    loss_func = torch.nn.MSELoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(args.epoch):
        model.train()
        for x, y in dataloader:
            if args.use_gpu:
                x = x.to(atorch.local_rank())
                y = y.to(atorch.local_rank())

            output = model(x)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Finish epoch {}/{}".format(epoch + 1, args.epoch))


if __name__ == "__main__":
    args = parse_args()
    main(args)
