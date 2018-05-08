# DistributedDataParallel

The folder contains demo scripts that run DistributedDataParallel with both native PyTorch syntax and `torchx.DistributedManager`. You can compare them and see the convenience of our wrapper. It's much less boilerplate code on users' side.

## native

```bash
chmod +x run_native.sh && ./run_native.sh
```

OR

```
python -m torch.distributed.launch --nproc_per_node=4 native.py
```

The experiment is launched with the new torch.distributed.launch utility introduced in v0.4

- https://pytorch.org/docs/stable/distributed.html#launch-utility
- https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

Warnings:
If you run with `gloo` backend, you'll see this issue
https://github.com/pytorch/pytorch/issues/2530
It doesn't affect runtime but prints error msg

DatasetSampler for DDP:

https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler

Official example on ImageNet distributed training:

https://github.com/pytorch/examples/blob/master/imagenet/main.py


## torchx.DistributedManager

Simply run 

```bash
python tx_manager.py
```

No external launcher! The `manager.entry()` call acts like `os.fork()`. It starts the worker processes and waits for them to finish, before terminating itself.

Run `watch nvidia-smi` to monitor the GPU usage.

Run assertion tests: the gradient should be the samed (all-reduced) across all GPUs unless `--disable-distributed`. CPU mode uses `gloo` backend while GPU mode uses `nccl` backend.

```bash
python test_tx_manager.py [--disable-distributed] [--use-cpu]
```

