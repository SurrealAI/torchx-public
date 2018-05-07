"""
Demo v0.4 NCCL multi-GPU collectives
https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions
https://pytorch.org/docs/stable/distributed.html
"""
import torch
import torch.distributed as dist
import torchx as tx

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
    local_gpu_ids = [0, 1]  # mean 1.0, mean 2.0
else:
    local_gpu_ids = [2, 3] # mean 3.0, mean 4.0


def new_tensor(device_id, value):
    with tx.device_scope(device_id, dtype=torch.float64):
        return torch.ones(SHAPE) * value


def get_tensor_list():
    return [new_tensor(device_id, value=device_id+1)
            for device_id in local_gpu_ids]

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
# all_gather semantics is quite complicated:
# https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_multigpu

tensor_list = get_tensor_list()

"""
Process 0: physical GPU 0, 1, output list device residence 
    [[gpu0, gpu0, gpu0, gpu0], [gpu1, gpu1, gpu1, gpu1]]
    values [[1., 2., 3., 4.], [1., 2., 3., 4.]]  from all GPUs across all procs
Process 1: physical GPU 2, 3, output list device residence 
    [[gpu2, gpu2, gpu2, gpu2], [gpu3, gpu3, gpu3, gpu3]]
    values [[1., 2., 3., 4.], [1., 2., 3., 4.]]  from all GPUs across all procs
"""
output_tensor_lists = [[new_tensor(i, value=0.) for _ in range(4)]
                       for i in local_gpu_ids]

dist.all_gather_multigpu(
    output_tensor_lists,
    tensor_list,
)
print('all_gather_multigpu rank '+str(local_rank) + '\n'
      + '\n\t'.join(map(str, output_tensor_lists)) )

for same_gpu_tensor_list in output_tensor_lists:
    assert_mean(same_gpu_tensor_list[0], 1.)
    assert_mean(same_gpu_tensor_list[1], 2.)
    assert_mean(same_gpu_tensor_list[2], 3.)
    assert_mean(same_gpu_tensor_list[3], 4.)

