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
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import torchx as tx


manager = tx.DistributedManager(
    None,  # backend will be set after we know CPU or GPU from command line arg
    num_procs=4,  # number of procs per node (1 node here)
)
manager.parser.add_argument('-d', '--disable-distributed', action='store_true')
manager.parser.add_argument('-c', '--use-cpu', action='store_true')
args = manager.parse_args()
backend = 'gloo' if args.use_cpu else 'nccl'
manager.set_backend(backend)
print('Backend:', backend)

manager.entry()

local_rank = manager.local_rank()

# when False, you will see the gradient to be different for each process
DISTRIBUTED = not args.disable_distributed


def get_grad_means(net):
    "returns scalars"
    return [w.grad.mean().item() for w
            in [net.fc1.weight, net.fc2.weight, net.fc3.weight]]


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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


INPUT_SIZE = (16, 10)

with manager.device_scope(use_cpu=args.use_cpu):
    fill_value = 1. * (local_rank + 1)
    x = torch.empty(0).new_full(INPUT_SIZE, fill_value)
    model = MyNet(10, 7, 5)
    if DISTRIBUTED:
        wrapped_model = tx.DistributedDataParallel(model)
    else:
        wrapped_model = model
    y = wrapped_model(x)
    y.backward(torch.ones_like(y))

# when DISTRIBUTED is True, you should see all grad means to be the same value
print(
    '\nRank {}\n\ty size {} value {} device {}'
    '\n\tw_grads {}'
    .format(local_rank, y.size(), y.mean(), y.device, get_grad_means(model))
)

if DISTRIBUTED:
    assert get_grad_means(model) == [1400., 2000., 2800.]
    print('assertion tests pass')
else:
    pass
