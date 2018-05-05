"""
DistributedDataParallel

Run with the new torch.distributed.launch util introduced in v0.4
https://pytorch.org/docs/stable/distributed.html#launch-utility

Warnings:
If you run with `gloo` backend, you'll see this issue
https://github.com/pytorch/pytorch/issues/2530
It doesn't affect runtime but prints error msg

python -m torch.distributed.launch --nproc_per_node=4 demo_ddp.py

DatasetSampler for DDP:
https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

Complete working example:
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import torchx as tx


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()


# WARNING: `gloo` backend will print error msg that wouldn't affect runtime
# https://github.com/pytorch/pytorch/issues/2530
dist.init_process_group(
    backend='nccl',
    # world_size=4,
)


DISTRIBUTED = True
weights = []


def get_grad_means():
    return [w.grad.mean() for w in weights]


class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.parallel.DistributedDataParallel(self.fc2)
        self.fc3 = nn.Linear(hidden_size, output_size)
        with torch.no_grad():
            for fc in [self.fc1, self.fc2, self.fc3]:
                fc.weight.fill_(1.)
                fc.bias.fill_(0.)
                global weights
                weights.append(fc.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


INPUT_SIZE = (16, 10)

with tx.device_scope(args.local_rank):
    fill_value = 1. * (args.local_rank + 1)
    x = torch.empty(0).new_full(INPUT_SIZE, fill_value)
    model = MyNet(10, 7, 5)
    if DISTRIBUTED:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    y = model(x)
    y.backward(torch.ones_like(y))

print('y', y.size(), 'mean', y.mean(),
      '\nw_grad', get_grad_means())

