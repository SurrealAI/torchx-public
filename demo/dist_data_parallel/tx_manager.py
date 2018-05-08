"""
DistributedDataParallel with TorchX DistributedManager

Warnings:
If you run with `gloo` backend, you'll see this issue
https://github.com/pytorch/pytorch/issues/2530
It doesn't affect runtime but prints error msg

python tx_dist_data_parallel.py

DatasetSampler for DDP:
https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

Complete working example:
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchx as tx

USE_CPU = False


manager = tx.DistributedManager(
    'gloo' if USE_CPU else 'nccl',
    num_procs=4,  # number of procs per node (1 node here)
)
# you can add your own command line args
manager.parser.add_argument('--my-flag', default=42)
manager.entry()

local_rank = manager.local_rank()
# get the parsed args after entry() call
args = manager.parse_args()
assert args.my_flag == 42


def get_grad_means(net):
    "returns scalars"
    return [w.grad.mean() for w
            in [net.fc1.weight, net.fc2.weight, net.fc3.weight]]


class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


INPUT_SIZE = (16, 10)

with manager.device_scope(use_cpu=USE_CPU):  # convenient scoping
    fill_value = 1. * (local_rank + 1)
    x = torch.empty(0).new_full(INPUT_SIZE, fill_value)
    model = MyNet(10, 7, 5)
    model_wrapped = tx.DistributedDataParallel(model)
    y = model_wrapped(x)
    y.backward(torch.ones_like(y))

# you should see all grad means to be the same value across different ranks
print(
    '\nRank {}\n\ty size {} value {} device {}'
    '\n\tw_grads {}'
    .format(local_rank, y.size(), y.mean(), y.device, get_grad_means(model))
)

