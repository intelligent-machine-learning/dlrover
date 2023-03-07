import torch

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    

def demo_basic(rank=None, world_size=None):
    if os.environ.get("distribute", False):
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size)
    # create model and move it to GPU with id rank
    model = LinearModel() 
    if os.environ.get("distribute", False):
        model = DDP(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(100):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss_fn(outputs, y_data).backward()
        optimizer.step()
        print(outputs)
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__=="__main__":
    if os.environ.get("distribute", False):
        run_demo(demo_basic,2)
    else:
        demo_basic()



 