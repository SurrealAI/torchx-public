"""
Demo collective functions for tcp, gloo, and nccl backends
https://pytorch.org/docs/stable/distributed.html#collective-functions
https://pytorch.org/docs/stable/distributed.html
"""
import torch
import torch.distributed as dist
import torchx as tx

# gloo backend doesn't work for some of the collectives, but works on both CPU/GPU
# WARNING: gloo will display harmless error messages, results are still correct
# nccl only works on GPU, but supports all of the collectives
# tcp only works on CPU, but supports all of the collectives

VALID_COMBOS = [
    ('tcp', 'cpu'),
    ('gloo', 'cpu'),
    ('gloo', 'gpu'),
    ('nccl', 'gpu'),
]

YOUR_CHOICE = 3
backend, mode = VALID_COMBOS[YOUR_CHOICE]


manager = tx.DistributedManager(
    backend,
    num_procs=4,
)
manager.entry()

local_rank = manager.local_rank()
print('LOCAL_RANK', local_rank)


if mode != 'cpu' and tx.gpu_count() >= 4 and backend in ['gloo', 'nccl']:
    device_id = local_rank
    print('RUNNING on GPU with', backend)
else:
    device_id = -1
    if mode != 'cpu' and tx.has_gpu():
        print('fewer than 4 GPUs, fall back to CPU with', backend)
    else:
        print('RUNNING on CPU with', backend)


def assert_mean(tensor, correct_mean):
    assert tensor.mean().item() == correct_mean

local_value = float(local_rank + 1)

SHAPE = (2, 3)


def new_tensor(device_id, value):
    with tx.device_scope(device_id, dtype=torch.float64):
        return torch.ones(SHAPE) * value


# ---------------- BROADCAST -----------------
if True:
    tensor = new_tensor(device_id, local_value)
    dist.broadcast(tensor, src=0)

    # all tensors become rank 0's value, 1.0
    print('{} AFTER broadcast {}'.format(local_rank, tensor))
    assert_mean(tensor, 1.)

# ---------------- REDUCE -----------------
if backend in ['tcp', 'nccl']:
    tensor = new_tensor(device_id, local_value)
    print('{} before {}'.format(local_rank, tensor))
    dist.reduce(tensor, dst=0, op=dist.reduce_op.SUM)

    # only rank 0 becomes 10.0
    print('{} AFTER reduce {}'.format(local_rank, tensor))
    if local_rank == 0:
        assert_mean(tensor, 10.)
    else:
        pass
        # somehow this doesn't hold, not sure why
        # assert_mean(tensor, local_value)

# ---------------- ALL_REDUCE -----------------
if True:
    tensor = new_tensor(device_id, local_value)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)

    # all ranks become 10.0
    print('{} AFTER all_reduce {}'.format(local_rank, tensor))
    assert_mean(tensor, 10.)

# ---------------- GATHER -----------------
if backend in ['tcp']:
    tensor = new_tensor(device_id, local_value)
    if local_rank == 0:
        gather_list = [new_tensor(device_id, 0.) for _ in range(4)]
        dist.gather(tensor, dst=0, gather_list=gather_list)
    else:
        dist.gather(tensor, dst=0)
        gather_list = None

    # all ranks become 24
    if local_rank == 0:
        print('{} AFTER gather {}'.format(local_rank, gather_list))
        assert_mean(gather_list[0], 1.)
        assert_mean(gather_list[1], 2.)
        assert_mean(gather_list[2], 3.)
        assert_mean(gather_list[3], 4.)

# ---------------- ALL_GATHER -----------------
if backend in ['tcp', 'nccl']:
    tensor = new_tensor(device_id, local_value)
    gather_list = [new_tensor(device_id, 0.) for _ in range(4)]
    dist.all_gather(gather_list, tensor)

    # all procs get a list of tensors 1.0 to 4.0
    print('{} AFTER all_gather {}'.format(local_rank, gather_list))
    assert_mean(gather_list[0], 1.)
    assert_mean(gather_list[1], 2.)
    assert_mean(gather_list[2], 3.)
    assert_mean(gather_list[3], 4.)

# ---------------- SCATTER -----------------
if backend in ['tcp']:
    tensor = new_tensor(device_id, 0.)
    if local_rank == 0:
        scatter_list = [new_tensor(device_id, i+1.0) for i in range(4)]
        dist.scatter(tensor, src=0, scatter_list=scatter_list)
    else:
        scatter_list = None
        dist.scatter(tensor, src=0)

    # all ranks become 24
    print('{} AFTER scatter {}'.format(local_rank, tensor))
    assert_mean(tensor, local_value)

