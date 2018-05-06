"""
Demo TCP backend point-to-point distributed communication
https://pytorch.org/docs/stable/distributed.html
"""
import torch
import torch.distributed as dist
import torchx as tx
import time

assert tx.gpu_count() >= 4, 'example only runs on >= 4 GPUs'

manager = tx.DistributedManager(
    'nccl',
    num_procs=2,  # number of procs per node (1 node here)
)
manager.entry()


def assert_mean(tensor, correct_mean):
    assert tensor.mean().item() == correct_mean

local_rank = manager.local_rank()
print('LOCAL_RANK', local_rank)

SHAPE = (2, 3)

# each tensor in the list from *each* process must reside on different GPUs
if local_rank == 0:
    gpu_ids = [0, 1]  # mean 1.0, mean 2.0
else:
    gpu_ids = [2, 3] # mean 4.0, mean 4.0


def get_tensor_list():
    tensor_list = []
    for device_id in gpu_ids:
        with tx.device_scope(device_id, dtype=torch.float64):
            tensor_list.append(torch.ones(SHAPE) * (device_id + 1))
    return tensor_list

# ---------------- ALL_REDUCE -----------------
tensor_list = get_tensor_list()
print('INPUT', tensor_list)

dist.all_reduce_multigpu(
    tensor_list,
    op=dist.reduce_op.PRODUCT
)

# all tensors become 1*2*3*4
print('{} AFTER all_reduce_multigpu {}'.format(local_rank, tensor_list))
if local_rank == 0:
    assert_mean(tensor_list[0], 24.)
    assert_mean(tensor_list[1], 24.)
else:
    assert_mean(tensor_list[0], 24.)
    assert_mean(tensor_list[1], 24.)

# ---------------- REDUCE -----------------
tensor_list = get_tensor_list()

# Only the GPU of tensor_list[0] on the process with rank dst is going to
# receive the final result.
dist.reduce_multigpu(
    tensor_list,
    dst=1,  # destination process rank
    op=dist.reduce_op.PRODUCT
)
print('{} AFTER reduce_multigpu {}'.format(local_rank, tensor_list))
if local_rank == 0:
    assert_mean(tensor_list[0], 1.)
    assert_mean(tensor_list[1], 2.)
else:
    assert_mean(tensor_list[0], 24.)
    assert_mean(tensor_list[1], 4.)

# ---------------- BROADCAST -----------------
tensor_list = get_tensor_list()

dist.broadcast_multigpu(
    tensor_list,
    src=1,  # rank 1 tensor_list[0] broadcast to all
)
print('{} AFTER broadcast_multigpu {}'.format(local_rank, tensor_list))
if local_rank == 0:
    assert_mean(tensor_list[0], 3.)
    assert_mean(tensor_list[1], 3.)
else:
    assert_mean(tensor_list[0], 3.)
    assert_mean(tensor_list[1], 3.)

# ---------------- ALL_GATHER -----------------
tensor_list = get_tensor_list()

dist.broadcast_multigpu(
    tensor_list,
    src=1,  # rank 1 tensor_list[0] broadcast to all
)
print('{} AFTER broadcast_multigpu {}'.format(local_rank, tensor_list))
if local_rank == 0:
    assert_mean(tensor_list[0], 3.)
    assert_mean(tensor_list[1], 3.)
else:
    assert_mean(tensor_list[0], 3.)
    assert_mean(tensor_list[1], 3.)
