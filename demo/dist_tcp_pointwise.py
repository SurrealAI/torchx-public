"""
Demo TCP backend point-to-point distributed communication
https://pytorch.org/docs/stable/distributed.html

In this example,
@1 broadcasts tensor [3.0], @0 and @2 will receive it in `x_b`
@0 sends tensor [5.0] to @1
@1 sends tensor [7.0] to @0
"""
import torch
import torch.distributed as dist
import torchx as tx
import time


def assert_mean(tensor, correct_mean):
    assert tensor.mean().item() == correct_mean


manager = tx.DistributedManager(
    'tcp',
    num_procs=3,  # number of procs per node (1 node here)
)
manager.entry()

local_rank = manager.local_rank()
print('LOCAL_RANK', local_rank)

BROADCAST_SHAPE = (2, 4)

if local_rank == 1:
    x_b = torch.ones(BROADCAST_SHAPE) * 3
else:
    x_b = torch.zeros(BROADCAST_SHAPE)
print('{} broadcast original {}'.format(local_rank, x_b))

# x_b is broadcasted when src=rank, otherwise it's the _receiver_
dist.broadcast(x_b, src=1)

assert_mean(x_b, 3.0)

SHAPE = (3, 3)

if local_rank == 0:
    print('0 broadcast recv', x_b)
    t1 = torch.zeros(SHAPE)
    sender_rank = dist.recv(t1)
    print('0 recv', t1, 'from sender', sender_rank)
    assert_mean(t1, 7.0)
    t0 = torch.ones(SHAPE) * 5
    dist.send(t0, 1)
elif local_rank == 1:
    print('1 broadcast send', x_b)
    t1 = torch.ones(SHAPE) * 7
    dist.send(t1, 0)  # send to rank 0
    t0 = torch.zeros(SHAPE)
    sender_rank = dist.recv(t0)
    print('1 recv', t0, 'from sender', sender_rank)
    assert_mean(t0, 5.0)
elif local_rank == 2:
    print('2 broadcast recv', x_b)
else:
    print('nothing')

