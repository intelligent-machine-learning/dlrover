# flake8: noqa: E402
import os
import time

import torch

local_rank = int(os.environ.get("LOCAL_RANK", "0"))

print(local_rank)

time.sleep(5)
tensor = torch.ones((10, 10), device=local_rank).bfloat16()

while True:
    time.sleep(0.2)
    torch.cuda.empty_cache()
    res = torch.matmul(tensor, tensor)
    tensor = tensor.cpu()
    tensor = tensor.cuda()
